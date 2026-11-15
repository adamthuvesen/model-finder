from __future__ import annotations

TARGET_COLUMN = "converted"
TIME_COLUMN = "event_date"

NUMERIC_COLUMNS = ["sessions_7d", "pages_viewed", "days_active"]
CATEGORICAL_COLUMNS = ["segment"]


def feature_columns() -> list[str]:
    return list(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
