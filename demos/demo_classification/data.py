from __future__ import annotations

TARGET_COLUMN = "will_expand_90d"
TIME_COLUMN = "snapshot_date"

BASE_NUMERIC_COLUMNS = [
    "account_age_days",
    "seat_count",
    "avg_weekly_active_users",
    "usage_growth_90d",
    "feature_adoption_count",
    "support_tickets_90d",
    "nps_score",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "seat_utilization",
    "support_per_active_user",
    "momentum_signal",
    "interaction_exec_growth",
]

CATEGORICAL_COLUMNS = ["region", "industry", "plan_tier"]
BOOLEAN_COLUMNS = ["has_exec_sponsor", "is_sales_touched", "previous_expansion"]


def base_feature_columns() -> list[str]:
    return BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS


def engineered_feature_columns() -> list[str]:
    return base_feature_columns() + ENGINEERED_NUMERIC_COLUMNS
