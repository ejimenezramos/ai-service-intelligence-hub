import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from html import unescape

from google import genai
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
logger = logging.getLogger(__name__)


DEFAULT_HUGGING_FACE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class AIQuotaExceededError(Exception):
    pass


class AlternativeAIError(Exception):
    pass


def get_hugging_face_config() -> tuple[str | None, str]:
    api_token = os.getenv("INTELLIGENCE_HUB_HUGGING_FACE_TOKEN")
    model_id = os.getenv("INTELLIGENCE_HUB_HUGGING_FACE_MODEL_ID") or DEFAULT_HUGGING_FACE_MODEL_ID
    return api_token, model_id


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY_2") 

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. Add GEMINI_API_KEY_2 to your .env file."
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

    raise ValueError("Gemini response could not be parsed as JSON.")


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
    context = build_incident_context(df)
    prompt_template = read_prompt_template("executive_summary_prompt.txt")
    prompt = prompt_template.replace("{{INCIDENT_CONTEXT}}", context)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return extract_json_from_response(response.text)

    except Exception as error:
        error_text = str(error).lower()

        if "quota" in error_text or "429" in error_text:
            raise AIQuotaExceededError(
                "Gemini quota exceeded. The app will continue using rule-based intelligence."
            ) from error

        raise


def generate_huggingface_decision_layer(df: pd.DataFrame) -> dict:
    api_token, model_id = get_hugging_face_config()

    if not api_token:
        raise AlternativeAIError("Hugging Face API token is missing.")

    context = build_incident_context(df)
    prompt_template = read_prompt_template("executive_summary_prompt.txt")
    prompt = prompt_template.replace("{{INCIDENT_CONTEXT}}", context)
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Do not include markdown or commentary.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1400,
            "temperature": 0.2,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        "https://router.huggingface.co/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw_response = response.read().decode("utf-8")
            parsed_response = json.loads(raw_response)

        choices = parsed_response.get("choices", []) if isinstance(parsed_response, dict) else []
        generated_text = choices[0].get("message", {}).get("content", "") if choices else ""

        if not generated_text:
            raise AlternativeAIError("Hugging Face returned an empty response.")

        return extract_json_from_response(generated_text)

    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as error:
        logger.exception("Hugging Face analysis failed.")
        raise AlternativeAIError("Hugging Face analysis failed.") from error
