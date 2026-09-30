import re
from io import StringIO
from zipfile import BadZipFile

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

_COLUMN_ALIASES = {
    "prioritu": "priority",
}


def normalize_column_name(column: object) -> str:
    """Normalize headers exported by spreadsheet apps on desktop or mobile."""
    normalized = str(column).strip().lower().strip("'\"`")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return _COLUMN_ALIASES.get(normalized, normalized)


def recover_mobile_csv_workbook(df: pd.DataFrame) -> pd.DataFrame:
    """Recover a CSV imported into Excel with the wrong regional delimiter."""
    if len(df.columns) > 4 or not any("," in str(value) for value in df.columns):
        return df

    # The remaining pandas column names are Unnamed placeholders because the
    # malformed workbook only has the real CSV header in its first cell.
    lines = [str(df.columns[0]).lstrip("'")]
    for row in df.itertuples(index=False, name=None):
        values = [str(value) for value in row if pd.notna(value)]
        if not values:
            continue
        # Excel treats a leading apostrophe as a text marker when importing a
        # CSV. It is not part of the original header or incident id.
        lines.append(";".join(values).lstrip("'"))

    if not lines:
        return df

    return pd.read_csv(StringIO("\n".join(lines)))


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
        try:
            df = pd.read_excel(file)
        except (BadZipFile, OSError, ValueError):
            # Some mobile spreadsheet apps preserve the CSV bytes while
            # changing the filename to .xlsx. Accept that common case.
            file.seek(0)
            df = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    df = recover_mobile_csv_workbook(df)
    df.columns = [normalize_column_name(column) for column in df.columns]
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
