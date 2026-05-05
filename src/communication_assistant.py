from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from urllib.parse import urlencode

import pandas as pd


STAKEHOLDER_PROFILES = [
    {
        "audience": "Technical team",
        "role": "technical owners and support engineers",
        "focus": "diagnosis, ownership, mitigation plan and evidence needed for permanent fixes",
    },
    {
        "audience": "Product / Service Owner",
        "role": "service owner or product owner",
        "focus": "customer impact, service stability, prioritization and expected business value",
    },
    {
        "audience": "Business stakeholder",
        "role": "affected business representative",
        "focus": "business impact, expected recovery, decision points and communication cadence",
    },
    {
        "audience": "Service Manager",
        "role": "service manager or incident process owner",
        "focus": "SLA risk, recurring pattern reduction, escalation path and governance actions",
    },
]


def _top_values(df: pd.DataFrame, column: str, limit: int = 3) -> list[str]:
    if column not in df.columns or df.empty:
        return []

    return [
        str(value)
        for value in df[column].dropna().astype(str).value_counts().head(limit).index.tolist()
        if str(value).strip()
    ]


def _join_items(items: list[str], fallback: str = "Not enough signal available") -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {item}" for item in cleaned) if cleaned else f"- {fallback}"


def _split_emails(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    candidates = re.split(r"[;,]", str(value))
    emails = []
    for candidate in candidates:
        email = candidate.strip()
        if email and "@" in email:
            emails.append(email)

    return emails


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _title_from_email(email: str) -> str:
    local_part = email.split("@", 1)[0]
    return local_part.replace(".", " ").replace("-", " ").title()


def _summarize_recipients(recipients: list[str], limit: int = 4) -> str:
    names = [_title_from_email(email) for email in _dedupe(recipients)]
    if not names:
        return "Recipients to be confirmed"

    visible = names[:limit]
    remaining = len(names) - len(visible)
    suffix = f" +{remaining} more" if remaining > 0 else ""
    return ", ".join(visible) + suffix


def _impact_summary_from_row(row: pd.Series | None) -> str:
    if row is None:
        return "Impact to be confirmed from the incident analysis"

    priority = str(row.get("priority", "N/A"))
    impact = str(row.get("estimated_impact", "N/A"))
    service = str(row.get("service", "impacted service"))
    recurrence = str(row.get("recurrence_type", ""))
    incident_id = str(row.get("incident_id", ""))
    recurrence_note = "Recurring signal" if "recurring" in recurrence.lower() else "Single incident signal"
    return f"{priority} priority | {impact} impact | {service} | {recurrence_note} | {incident_id}"


def _impact_rank(row: pd.Series) -> int:
    priority = str(row.get("priority", "")).lower()
    impact = str(row.get("estimated_impact", "")).lower()
    backlog_priority = str(row.get("suggested_backlog_priority", "")).lower()
    recurrence = str(row.get("recurrence_type", "")).lower()

    score = 0
    if priority == "critical":
        score += 40
    elif priority == "high":
        score += 30
    elif priority == "medium":
        score += 15

    if impact == "high":
        score += 30
    elif impact == "medium-high":
        score += 20

    if backlog_priority.startswith("p1"):
        score += 20
    elif backlog_priority.startswith("p2"):
        score += 10

    if "recurring" in recurrence:
        score += 15

    return score


def _stakeholder_emails_for_row(row: pd.Series) -> list[str]:
    emails = []
    for column in ["stakeholder_emails", "team_emails", "opened_by_email"]:
        emails.extend(_split_emails(row.get(column)))

    return _dedupe(emails)


def _top_incident_rows(enriched_df: pd.DataFrame, limit: int = 2) -> pd.DataFrame:
    if enriched_df.empty:
        return enriched_df

    ranked = enriched_df.copy()
    ranked["_assistant_impact_rank"] = ranked.apply(_impact_rank, axis=1)
    return ranked.sort_values(
        by=["_assistant_impact_rank", "value_score"],
        ascending=[False, False],
    ).head(limit)


def _build_context(enriched_df: pd.DataFrame, ai_result: dict, ai_mode: str) -> dict:
    top_services = _top_values(enriched_df, "service")
    top_root_causes = _top_values(enriched_df, "probable_root_cause")
    assignment_groups = _top_values(enriched_df, "assignment_group")
    assigned_people = _top_values(enriched_df, "assigned_to")

    primary_service = top_services[0] if top_services else "the impacted services"
    leadership_actions = ai_result.get("leadership_actions", [])

    return {
        "ai_mode": ai_mode,
        "primary_service": primary_service,
        "top_services": top_services,
        "top_root_causes": top_root_causes,
        "assignment_groups": assignment_groups,
        "assigned_people": assigned_people,
        "executive_summary": ai_result.get("executive_summary", "No executive summary returned."),
        "leadership_actions": leadership_actions if isinstance(leadership_actions, list) else [str(leadership_actions)],
        "stakeholder_message": ai_result.get("stakeholder_message", ""),
    }


def _email_link(base_url: str, params: dict[str, str]) -> str:
    return f"{base_url}?{urlencode(params)}"


def _build_email_body(profile: dict, context: dict, row: pd.Series | None = None) -> str:
    service = context["primary_service"]
    incident_id = "priority incident"
    business_impact = context["stakeholder_message"] or context["executive_summary"]
    recommended_action = profile["focus"]
    priority = "priority"
    impact = "business"

    if row is not None:
        service = str(row.get("service", service))
        incident_id = str(row.get("incident_id", incident_id))
        business_impact = str(row.get("business_impact", business_impact))
        recommended_action = str(row.get("recommended_action", recommended_action))
        priority = str(row.get("priority", priority)).lower()
        impact = str(row.get("estimated_impact", impact)).lower()

    return f"""Hi,

I am reaching out following the latest AI Service Intelligence review for {service}. The analysis has highlighted {incident_id} as a {priority}-priority item with {impact} impact, and it requires coordinated follow-up across the relevant business and technical stakeholders.

Why this matters
{business_impact}

Executive context
{context["executive_summary"]}

Recommended leadership actions
{_join_items(context["leadership_actions"])}

Requested focus for {profile["role"]}
- {profile["focus"]}
- {recommended_action}

Operational signals to consider
Services: {", ".join(context["top_services"]) or "N/A"}
Probable root causes: {", ".join(context["top_root_causes"]) or "N/A"}
Potential owners: {", ".join(context["assignment_groups"]) or "N/A"}

Could you please review the proposed actions and confirm:
- accountable owner for the next step
- any business constraints or timing risks
- whether escalation or backlog prioritization is required

I will consolidate the feedback and keep the action plan aligned with the incident priorities.

Best regards,
AI Service Intelligence Hub"""


def _email_record(
    audience: str,
    mode: str,
    subject: str,
    body: str,
    recipients: list[str],
    rationale: str = "",
    stakeholder_summary: str = "",
    impact_summary: str = "",
) -> dict:
    to = ",".join(_dedupe(recipients))

    return {
        "audience": audience,
        "mode": mode,
        "subject": subject,
        "content": body,
        "recipients": to or "Add recipients before sending",
        "stakeholder_summary": stakeholder_summary or _summarize_recipients(recipients),
        "impact_summary": impact_summary or "Impact derived from the AI decision brief",
        "rationale": rationale,
        "gmail_link": _email_link(
            "https://mail.google.com/mail/",
            {
                "view": "cm",
                "fs": "1",
                "to": to,
                "su": subject,
                "body": body,
            },
        ),
        "outlook_link": _email_link(
            "https://outlook.live.com/mail/0/deeplink/compose",
            {
                "to": to,
                "subject": subject,
                "body": body,
            },
        ),
    }


def _emails_from_ai_suggestions(ai_result: dict, ai_mode: str) -> list[dict]:
    suggestions = ai_result.get("email_suggestions", [])
    if not isinstance(suggestions, list):
        return []

    emails = []
    for suggestion in suggestions[:2]:
        if not isinstance(suggestion, dict):
            continue

        subject = str(suggestion.get("subject", "")).strip()
        body = str(suggestion.get("body", "")).strip()
        recipients = suggestion.get("recipients", [])

        if isinstance(recipients, str):
            recipients = _split_emails(recipients)
        elif isinstance(recipients, list):
            recipients = _dedupe([email for value in recipients for email in _split_emails(value)])
        else:
            recipients = []

        if not subject or not body:
            continue

        emails.append(
            _email_record(
                audience=str(suggestion.get("audience", "Stakeholder follow-up")),
                mode=ai_mode,
                subject=subject,
                body=body,
                recipients=recipients,
                rationale=str(suggestion.get("rationale", "")),
                stakeholder_summary=_summarize_recipients(recipients),
                impact_summary=str(suggestion.get("rationale", "Based on AI-prioritized incident impact"))[:180],
            )
        )

    return emails


def generate_all_emails(enriched_df: pd.DataFrame, ai_result: dict, ai_mode: str) -> list[dict]:
    ai_emails = _emails_from_ai_suggestions(ai_result, ai_mode)
    if ai_emails:
        return ai_emails

    context = _build_context(enriched_df, ai_result, ai_mode)
    emails = []
    top_rows = _top_incident_rows(enriched_df)

    for index, profile in enumerate(STAKEHOLDER_PROFILES):
        row = top_rows.iloc[index] if index < len(top_rows) else None
        service = str(row.get("service", context["primary_service"])) if row is not None else context["primary_service"]
        incident_id = str(row.get("incident_id", "")) if row is not None else ""
        recipients = _stakeholder_emails_for_row(row) if row is not None else []
        subject = f"Incident improvement follow-up: {service}"
        body = _build_email_body(profile, context, row)

        emails.append(
            _email_record(
                audience=profile["audience"],
                mode=ai_mode,
                subject=subject,
                body=body,
                recipients=recipients,
                rationale="Fallback suggestion based on incident priority, impact, recurrence and stakeholder email columns.",
                stakeholder_summary=_summarize_recipients(recipients),
                impact_summary=_impact_summary_from_row(row),
            )
        )

    return emails[:2]


def _calendar_dates() -> str:
    start_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_time = start_time + timedelta(days=1, hours=1)
    end_time = start_time + timedelta(minutes=45)
    return f"{start_time:%Y%m%dT%H%M%SZ}/{end_time:%Y%m%dT%H%M%SZ}"


def _google_calendar_link(title: str, description: str) -> str:
    return _email_link(
        "https://calendar.google.com/calendar/render",
        {
            "action": "TEMPLATE",
            "text": title,
            "details": description,
            "dates": _calendar_dates(),
        },
    )


def _build_call_description(purpose: str, context: dict, stakeholders: list[str]) -> str:
    return f"""Purpose
{purpose}

Suggested stakeholders
{_join_items(stakeholders, "Add the relevant stakeholder emails before sending the invite")}

Context
{context["executive_summary"]}

Leadership actions to review
{_join_items(context["leadership_actions"])}

Operational signals
Services: {", ".join(context["top_services"]) or "N/A"}
Probable root causes: {", ".join(context["top_root_causes"]) or "N/A"}
Potential owners: {", ".join(context["assignment_groups"]) or "N/A"}
Named owners in sample: {", ".join(context["assigned_people"]) or "N/A"}"""


def generate_call_suggestions(enriched_df: pd.DataFrame, ai_result: dict, ai_mode: str) -> list[dict]:
    context = _build_context(enriched_df, ai_result, ai_mode)
    primary_service = context["primary_service"]

    suggestions = [
        {
            "title": f"Leadership alignment: {primary_service} stability actions",
            "purpose": "Align stakeholders on incident impact, ownership, decision points and immediate leadership actions.",
            "stakeholders": [
                "Service Manager",
                "Product / Service Owner",
                "Business stakeholder",
                "Technical team lead",
            ],
        },
        {
            "title": f"Technical review: {primary_service} recurring incidents",
            "purpose": "Review technical evidence, recurring root causes, monitoring gaps and permanent-fix candidates.",
            "stakeholders": [
                "Technical team",
                *context["assignment_groups"][:2],
            ],
        },
    ]

    calls = []
    for suggestion in suggestions:
        stakeholder_emails = []
        for _, row in _top_incident_rows(enriched_df, limit=2).iterrows():
            stakeholder_emails.extend(_stakeholder_emails_for_row(row))

        suggested_stakeholders = _dedupe([*suggestion["stakeholders"], *stakeholder_emails[:6]])
        description = _build_call_description(
            suggestion["purpose"],
            context,
            suggested_stakeholders,
        )
        calls.append(
            {
                "title": suggestion["title"],
                "description": description,
                "stakeholders": ", ".join(suggested_stakeholders),
                "stakeholder_summary": _summarize_recipients(stakeholder_emails[:6]),
                "purpose": suggestion["purpose"],
                "calendar_link": _google_calendar_link(suggestion["title"], description),
            }
        )

    return calls[:2]
