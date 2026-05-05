import pandas as pd

PRIORITY_SCORE = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 1,
}

EFFORT_SCORE = {
    "Low": 1,
    "Medium": 2,
    "Medium-High": 3,
    "High": 4,
}


def calculate_value_score(row: pd.Series) -> int:
    priority = PRIORITY_SCORE.get(str(row.get("priority", "")).lower(), 2)
    recurrence_bonus = 2 if row.get("recurrence_type") == "Recurring issue" else 0
    reopened_bonus = min(int(row.get("reopened_count", 0) or 0), 3)
    impact_bonus = 2 if row.get("estimated_impact") == "High" else 1

    return priority + recurrence_bonus + reopened_bonus + impact_bonus


def calculate_effort_score(row: pd.Series) -> int:
    return EFFORT_SCORE.get(row.get("estimated_effort", "Medium"), 2)


def add_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["value_score"] = scored.apply(calculate_value_score, axis=1)
    scored["effort_score"] = scored.apply(calculate_effort_score, axis=1)
    scored["value_effort_ratio"] = (scored["value_score"] / scored["effort_score"]).round(2)

    return scored.sort_values(
        by=["value_effort_ratio", "value_score"],
        ascending=[False, False],
    )