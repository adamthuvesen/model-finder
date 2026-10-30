from __future__ import annotations

TARGET_COLUMN = "will_convert"
TIME_COLUMN = "event_date"

NUMERIC_COLUMNS = ["signal_a", "signal_b", "signal_c", "signal_d"]
CATEGORICAL_COLUMNS = ["channel"]


def feature_columns() -> list[str]:
    return NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
