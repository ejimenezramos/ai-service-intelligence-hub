import pandas as pd

from src.prioritization import add_priority_scores
from src.executive_summary import build_rule_based_summary


def classify_recurrence(row: pd.Series) -> str:
    text = f"{row.get('short_description', '')} {row.get('description', '')}".lower()
    reopened = int(row.get("reopened_count", 0) or 0)

    recurrence_keywords = [
        "again", "repeated", "intermittent", "continues",
        "similar", "recurring", "delayed again", "failed again"
    ]

    if reopened > 0 or any(word in text for word in recurrence_keywords):
        return "Recurring issue"

    return "Single occurrence"


def infer_probable_root_cause(row: pd.Series) -> str:
    text = f"{row.get('short_description', '')} {row.get('description', '')}".lower()
    category = str(row.get("category", "")).lower()

    if "authentication" in text or "login" in text:
        return "Authentication / identity service instability"
    if "timeout" in text or "database" in text or category == "database":
        return "Performance or database query bottleneck"
    if "etl" in text or "data not refreshed" in text or "missing data" in text:
        return "Data pipeline or monitoring gap"
    if "access" in category or "role" in text:
        return "Access management / role mapping issue"
    if "certificate" in text:
        return "Infrastructure certificate lifecycle issue"
    if "vendor" in text or "file" in text:
        return "Integration or scheduled automation failure"
    if "synchronization" in text or "sync" in text:
        return "System integration failure"
    if "alert" in text or "monitoring" in category:
        return "Monitoring coverage gap"

    return "Requires further investigation"


def estimate_impact(row: pd.Series) -> str:
    priority = str(row.get("priority", "")).lower()
    reopened = int(row.get("reopened_count", 0) or 0)
    business_impact = str(row.get("business_impact", "")).lower()

    if priority == "critical" or reopened >= 2 or "stopped" in business_impact:
        return "High"
    if priority == "high" or "delayed" in business_impact or "degraded" in business_impact:
        return "Medium-High"
    if priority == "medium":
        return "Medium"

    return "Low"


def estimate_effort(row: pd.Series) -> str:
    root_cause = infer_probable_root_cause(row).lower()

    if "access" in root_cause:
        return "Low"
    if "monitoring" in root_cause or "scheduled automation" in root_cause:
        return "Medium"
    if "authentication" in root_cause or "integration" in root_cause:
        return "Medium-High"
    if "infrastructure" in root_cause or "database" in root_cause:
        return "Medium"

    return "Medium"


def suggest_backlog_priority(row: pd.Series) -> str:
    impact = estimate_impact(row)
    effort = estimate_effort(row)
    recurrence = classify_recurrence(row)

    if impact == "High" and recurrence == "Recurring issue":
        return "P1 - Immediate improvement candidate"
    if impact in ["High", "Medium-High"] and effort in ["Low", "Medium"]:
        return "P2 - High-value backlog candidate"
    if recurrence == "Recurring issue":
        return "P2 - Recurrence reduction candidate"
    if impact == "Medium":
        return "P3 - Monitor and refine"

    return "P4 - Low priority / operational handling"


def suggest_action(row: pd.Series) -> str:
    root_cause = infer_probable_root_cause(row).lower()

    if "authentication" in root_cause:
        return "Investigate connector stability, timeout thresholds, identity dependencies and monitoring coverage."
    if "database" in root_cause:
        return "Review query performance, report scheduling windows and database load during peak periods."
    if "data pipeline" in root_cause:
        return "Improve ETL monitoring, warning handling and alerts for incomplete executions."
    if "access" in root_cause:
        return "Standardize role mapping and create an access validation checklist for onboarding and role changes."
    if "certificate" in root_cause:
        return "Add certificate expiry tracking and proactive renewal alerts."
    if "scheduled automation" in root_cause:
        return "Review job scheduling, add failure alerts and document recovery steps."
    if "integration" in root_cause:
        return "Map integration dependencies and define clear escalation ownership between involved teams."
    if "monitoring" in root_cause:
        return "Review alert rules, ownership and monitoring coverage for failed jobs and degraded services."

    return "Clarify ownership, collect technical evidence and define a follow-up improvement action."


def enrich_incidents(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["recurrence_type"] = enriched.apply(classify_recurrence, axis=1)
    enriched["probable_root_cause"] = enriched.apply(infer_probable_root_cause, axis=1)
    enriched["estimated_impact"] = enriched.apply(estimate_impact, axis=1)
    enriched["estimated_effort"] = enriched.apply(estimate_effort, axis=1)
    enriched["suggested_backlog_priority"] = enriched.apply(suggest_backlog_priority, axis=1)
    enriched["recommended_action"] = enriched.apply(suggest_action, axis=1)

    return add_priority_scores(enriched)


def get_executive_summary(df: pd.DataFrame) -> dict:
    enriched = enrich_incidents(df)
    return build_rule_based_summary(enriched)


def get_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["service", "probable_root_cause", "recurrence_type"])
        .size()
        .reset_index(name="incident_count")
        .sort_values(by="incident_count", ascending=False)
    )