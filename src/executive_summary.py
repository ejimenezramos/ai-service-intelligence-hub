import pandas as pd


def build_rule_based_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "recurring_count": 0,
            "p1_count": 0,
            "top_service": "N/A",
            "top_root_cause": "N/A",
            "main_recommendation": "No incidents available for analysis.",
        }

    recurring_count = len(df[df["recurrence_type"] == "Recurring issue"])
    p1_count = len(df[df["suggested_backlog_priority"].str.startswith("P1", na=False)])

    top_service = df["service"].value_counts().idxmax()
    top_root_cause = df["probable_root_cause"].value_counts().idxmax()

    return {
        "recurring_count": recurring_count,
        "p1_count": p1_count,
        "top_service": top_service,
        "top_root_cause": top_root_cause,
        "main_recommendation": (
            f"Prioritize recurring issues in {top_service}, especially those related to "
            f"{top_root_cause.lower()}, as they show potential for reducing operational workload "
            f"and improving service stability."
        ),
    }
def build_fallback_ai_decision(df: pd.DataFrame) -> dict:
    summary = build_rule_based_summary(df)

    top_patterns = (
        df.groupby(["service", "probable_root_cause"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .head(3)
    )

    key_patterns = [
        f"{row['service']} shows repeated signals related to {row['probable_root_cause']}."
        for _, row in top_patterns.iterrows()
    ]

    backlog_items = (
        df.sort_values(by=["value_effort_ratio", "value_score"], ascending=[False, False])
        .head(3)
    )

    backlog_priorities = [
        f"{row['incident_id']} - {row['service']}: prioritize due to {row['estimated_impact']} impact and {row['estimated_effort']} effort."
        for _, row in backlog_items.iterrows()
    ]

    return {
        "executive_summary": summary["main_recommendation"],
        "key_patterns": key_patterns,
        "probable_root_causes": df["probable_root_cause"].value_counts().head(3).index.tolist(),
        "backlog_priorities": backlog_priorities,
        "leadership_actions": [
            "Review recurring services with the responsible teams.",
            "Convert high-value incidents into backlog improvement candidates.",
            "Align technical owners and business stakeholders around the highest-impact issues.",
        ],
        "stakeholder_message": (
            f"The main hotspot is {summary['top_service']}. "
            f"The recommended focus is reducing recurring incidents linked to {summary['top_root_cause']}."
        ),
    }