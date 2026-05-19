import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from html import unescape

from google import genai
from google.genai import types
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
logger = logging.getLogger(__name__)


DEFAULT_GEMINI_MODEL_ID = "gemini-2.5-flash-lite"
DEFAULT_HUGGING_FACE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct:novita"
DEFAULT_HUGGING_FACE_FALLBACK_MODEL_IDS = [
    "meta-llama/Llama-3.1-8B-Instruct:novita",
    "deepseek-ai/DeepSeek-V4-Flash:novita",
    "google/gemma-4-31B-it:novita",
]
LEGACY_UNSUPPORTED_HUGGING_FACE_MODEL_IDS = {"Qwen/Qwen2.5-1.5B-Instruct"}
UNSUPPORTED_HUGGING_FACE_MODEL_IDS = {
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-7B-Instruct",
    *LEGACY_UNSUPPORTED_HUGGING_FACE_MODEL_IDS,
}
HUGGING_FACE_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
GEMINI_MAX_OUTPUT_TOKENS = 2200
HUGGING_FACE_MAX_OUTPUT_TOKENS = 1800
AI_GENERATION_TEMPERATURE = 0.2


class AIQuotaExceededError(Exception):
    pass


class AlternativeAIError(Exception):
    def __init__(self, message: str, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category


def get_hugging_face_config() -> tuple[str | None, str]:
    api_token = os.getenv("INTELLIGENCE_HUB_HF_TOKEN") or os.getenv(
        "INTELLIGENCE_HUB_HUGGING_FACE_TOKEN"
    )
    new_model_id = os.getenv("INTELLIGENCE_HUB_HF_MODEL")
    legacy_model_id = os.getenv("INTELLIGENCE_HUB_HUGGING_FACE_MODEL_ID")
    model_id = new_model_id or legacy_model_id or DEFAULT_HUGGING_FACE_MODEL_ID

    if model_id in UNSUPPORTED_HUGGING_FACE_MODEL_IDS:
        logger.warning(
            "Hugging Face configured model skipped | category=model_not_supported | "
            "configured_model=%s | replacement_model=%s",
            model_id,
            DEFAULT_HUGGING_FACE_MODEL_ID,
        )
        model_id = DEFAULT_HUGGING_FACE_MODEL_ID

    return api_token, model_id


def _dedupe_model_ids(model_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(model_id.strip() for model_id in model_ids if model_id.strip()))


def get_hugging_face_model_candidates() -> list[str]:
    _, primary_model_id = get_hugging_face_config()
    configured_fallbacks = os.getenv("INTELLIGENCE_HUB_HF_FALLBACK_MODELS", "")
    fallback_model_ids = [
        model_id.strip()
        for model_id in configured_fallbacks.split(",")
        if model_id.strip() and model_id.strip() not in UNSUPPORTED_HUGGING_FACE_MODEL_IDS
    ] or DEFAULT_HUGGING_FACE_FALLBACK_MODEL_IDS

    return _dedupe_model_ids([primary_model_id, *fallback_model_ids])


def get_gemini_model_id() -> str:
    return (
        os.getenv("INTELLIGENCE_HUB_GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL_ID
    )


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("INTELLIGENCE_HUB_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_2")

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. Add INTELLIGENCE_HUB_GEMINI_API_KEY to your .env file."
        )

    return genai.Client(api_key=api_key)


def read_prompt_template(file_name: str) -> str:
    path = PROMPTS_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    return path.read_text(encoding="utf-8")


def _compact_text(value: object, max_chars: int = 180) -> str:
    text = str(value).replace("\n", " ").strip()
    return text[: max_chars - 3].rstrip() + "..." if len(text) > max_chars else text


def build_incident_context(df: pd.DataFrame, max_rows: int = 10) -> str:
    summary_lines = [
        f"Total incidents: {len(df)}",
    ]

    for column, label in [
        ("priority", "Priority distribution"),
        ("service", "Top services"),
        ("probable_root_cause", "Top probable root causes"),
        ("recurrence_type", "Recurrence distribution"),
        ("suggested_backlog_priority", "Backlog priority distribution"),
    ]:
        if column in df.columns:
            values = df[column].fillna("Unknown").astype(str).value_counts().head(5)
            summary = ", ".join(f"{name}: {count}" for name, count in values.items())
            summary_lines.append(f"{label}: {summary}")

    selected_columns = [
        "incident_id",
        "priority",
        "state",
        "service",
        "category",
        "short_description",
        "business_impact",
        "recurrence_type",
        "probable_root_cause",
        "estimated_impact",
        "estimated_effort",
        "suggested_backlog_priority",
        "value_score",
        "effort_score",
        "recommended_action",
    ]

    available_columns = [col for col in selected_columns if col in df.columns]
    sample = df.copy()
    sort_columns = [column for column in ["value_score", "reopened_count"] if column in sample.columns]
    if sort_columns:
        sample = sample.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
    sample = sample[available_columns].head(max_rows)
    for column in ["short_description", "business_impact", "recommended_action"]:
        if column in sample.columns:
            sample[column] = sample[column].apply(_compact_text)

    return (
        "Operational summary:\n"
        + "\n".join(f"- {line}" for line in summary_lines)
        + "\n\nPrioritized incident sample:\n"
        + sample.to_json(orient="records", force_ascii=False)
    )


def extract_json_from_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned)
    cleaned = re.sub(r"^```", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return sanitize_ai_payload(json.loads(cleaned))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return sanitize_ai_payload(json.loads(match.group(0)))

    raise ValueError("AI response could not be parsed as JSON.")


def sanitize_ai_text(value: str) -> str:
    text = unescape(str(value))
    text = re.sub(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def sanitize_ai_payload(value):
    if isinstance(value, dict):
        return {key: sanitize_ai_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_ai_payload(item) for item in value]
    if isinstance(value, str):
        return sanitize_ai_text(value)
    return value


def generate_ai_decision_layer(df: pd.DataFrame) -> dict:
    client = get_gemini_client()
    model_id = get_gemini_model_id()
    context = build_incident_context(df)
    prompt_template = read_prompt_template("executive_summary_prompt.txt")
    prompt = prompt_template.replace("{{INCIDENT_CONTEXT}}", context)

    generation_config = types.GenerateContentConfig(
        temperature=AI_GENERATION_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
    )

    for attempt in range(1, 3):
        try:
            logger.info("Gemini request | model=%s | attempt=%s", model_id, attempt)
            request_prompt = prompt
            if attempt > 1:
                request_prompt = (
                    prompt
                    + "\n\nFinal formatting reminder: return one complete, valid JSON object only. "
                    "Do not omit commas, brackets or required keys."
                )

            response = client.models.generate_content(
                model=model_id,
                contents=request_prompt,
                config=generation_config,
            )
            return extract_json_from_response(response.text)

        except Exception as error:
            category, detail = classify_gemini_error(error)
            logger.warning(
                "Gemini failed | model=%s | category=%s | attempt=%s | detail=%s",
                model_id,
                category,
                attempt,
                detail,
            )

            if category == "quota":
                raise AIQuotaExceededError(
                    "Gemini quota exceeded. The app will continue using rule-based intelligence."
                ) from error

            if category in {"availability", "response_parse"} and attempt == 1:
                time.sleep(1.25)
                continue

            raise

    raise RuntimeError("Gemini analysis failed without a captured provider error.")


def classify_gemini_error(error: Exception) -> tuple[str, str]:
    status_code = getattr(error, "status_code", None)
    text = str(error)
    text_lower = text.lower()

    if isinstance(error, (json.JSONDecodeError, ValueError)) or "could not be parsed as json" in text_lower:
        return "response_parse", _compact_text(text, max_chars=500)

    if status_code == 429 or "429" in text_lower or "quota" in text_lower or "resource_exhausted" in text_lower:
        return "quota", _compact_text(text, max_chars=500)

    if status_code == 503 or "503" in text_lower or "unavailable" in text_lower or "high demand" in text_lower:
        return "availability", _compact_text(text, max_chars=500)

    if status_code in {401, 403} or "api key" in text_lower or "permission" in text_lower:
        return "auth_or_permissions", _compact_text(text, max_chars=500)

    if status_code == 400 or "400" in text_lower or "bad request" in text_lower:
        return "request", _compact_text(text, max_chars=500)

    return "other", _compact_text(text, max_chars=500)


def _read_http_error(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def classify_hugging_face_error(status_code: int | None, response_body: str = "") -> str:
    text = response_body.lower()

    if status_code == 429 or "quota" in text or "rate limit" in text or "too many requests" in text:
        return "quota"

    if "cloudflare" in text or "access denied | api." in text:
        return "provider_access_denied"

    if status_code in {401, 403} or "unauthorized" in text or "forbidden" in text or "permission" in text:
        return "auth_or_permissions"

    if "model_not_supported" in text or "not supported by any provider" in text:
        return "model_not_supported"

    if status_code == 404 or "cannot post /models" in text or "not found" in text:
        return "model_or_endpoint_not_found"

    if status_code in {500, 502, 503, 504} or "unavailable" in text or "loading" in text:
        return "availability"

    if status_code == 400 or "bad request" in text or "invalid_request" in text:
        return "request"

    return "other"


def _request_json(url: str, payload: dict, api_token: str, timeout: int = 45) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_response = response.read().decode("utf-8")
        return json.loads(raw_response)


def _generate_hugging_face_router(prompt: str, model_id: str, api_token: str) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON. Do not include markdown or commentary.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": HUGGING_FACE_MAX_OUTPUT_TOKENS,
        "temperature": AI_GENERATION_TEMPERATURE,
        "response_format": {"type": "json_object"},
    }
    parsed_response = _request_json(HUGGING_FACE_ROUTER_URL, payload, api_token)
    choices = parsed_response.get("choices", []) if isinstance(parsed_response, dict) else []
    return choices[0].get("message", {}).get("content", "") if choices else ""


def _generate_hugging_face_text(prompt: str, model_id: str, api_token: str) -> str:
    try:
        return _generate_hugging_face_router(prompt, model_id, api_token)
    except urllib.error.HTTPError as router_error:
        router_body = _read_http_error(router_error)
        router_category = classify_hugging_face_error(router_error.code, router_body)
        logger.warning(
            "Hugging Face router failed | model=%s | category=%s | status=%s | response=%s",
            model_id,
            router_category,
            router_error.code,
            _compact_text(router_body, max_chars=700),
        )

        if router_category in {"auth_or_permissions", "quota"}:
            raise AlternativeAIError("Hugging Face router request failed.", category=router_category) from router_error

        raise AlternativeAIError("Hugging Face router request failed.", category=router_category) from router_error


def generate_huggingface_decision_layer(df: pd.DataFrame) -> dict:
    api_token, _ = get_hugging_face_config()

    if not api_token:
        logger.warning("Hugging Face skipped | category=auth_or_permissions | detail=token_missing")
        raise AlternativeAIError("Hugging Face API token is missing.", category="auth_or_permissions")

    context = build_incident_context(df)
    prompt_template = read_prompt_template("executive_summary_prompt.txt")
    prompt = prompt_template.replace("{{INCIDENT_CONTEXT}}", context)

    last_error_category = "unknown"
    for model_id in get_hugging_face_model_candidates():
        try:
            logger.info("Hugging Face request | model=%s", model_id)
            generated_text = _generate_hugging_face_text(prompt, model_id, api_token)

            if not generated_text:
                logger.warning("Hugging Face failed | category=empty_response | model=%s", model_id)
                last_error_category = "empty_response"
                continue

            return extract_json_from_response(generated_text)

        except urllib.error.HTTPError as error:
            body = _read_http_error(error)
            category = classify_hugging_face_error(error.code, body)
            logger.warning(
                "Hugging Face inference failed | model=%s | category=%s | status=%s | response=%s",
                model_id,
                category,
                error.code,
                _compact_text(body, max_chars=700),
            )
            last_error_category = category
            if category in {"auth_or_permissions", "quota"}:
                raise AlternativeAIError("Hugging Face analysis failed.", category=category) from error

        except (urllib.error.URLError, json.JSONDecodeError, ValueError, AlternativeAIError) as error:
            category = getattr(error, "category", "response_or_network")
            logger.warning(
                "Hugging Face failed | model=%s | category=%s | detail=%s",
                model_id,
                category,
                _compact_text(error, max_chars=500),
            )
            last_error_category = category
            if category in {"auth_or_permissions", "quota"}:
                raise AlternativeAIError("Hugging Face analysis failed.", category=category) from error

    raise AlternativeAIError("Hugging Face analysis failed.", category=last_error_category)
