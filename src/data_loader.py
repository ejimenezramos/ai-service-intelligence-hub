import pandas as pd


REQUIRED_COLUMNS = [
    "incident_id",
    "opened_at",
    "priority",
    "state",
    "assignment_group",
    "service",
    "category",
    "short_description",
    "description",
    "business_impact",
]


def load_incidents(file) -> pd.DataFrame:
    file_name = str(getattr(file, "name", "")).strip()
    file_suffix = file_name.lower()

    # Mobile file pickers may preserve an uppercase extension or return a
    # file-like object whose cursor was already read by the uploader.
    if hasattr(file, "seek"):
        file.seek(0)

    if file_suffix.endswith(".csv"):
        df = pd.read_csv(file)
    elif file_suffix.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce")

    if "closed_at" in df.columns:
        df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")

    return df


def calculate_basic_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    open_incidents = len(df[df["state"].str.lower().isin(["open", "in progress"])])
    critical_high = len(df[df["priority"].str.lower().isin(["critical", "high"])])
    reopened = df["reopened_count"].sum() if "reopened_count" in df.columns else 0

    return {
        "total": total,
        "open_incidents": open_incidents,
        "critical_high": critical_high,
        "reopened": int(reopened),
    }
