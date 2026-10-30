from __future__ import annotations

TARGET_COLUMN = "net_revenue_change_180d"
TIME_COLUMN = "snapshot_date"

BASE_NUMERIC_COLUMNS = [
    "account_age_days",
    "seat_count",
    "avg_weekly_active_users",
    "usage_growth_90d",
    "feature_adoption_count",
    "champion_engagement_score",
    "support_tickets_90d",
    "nps_score",
    "contract_value_t0",
    "renewal_window_days",
    "previous_revenue_change_365d",
    "marketing_touchpoints_90d",
    "webinar_attendance_90d",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "seat_utilization",
    "support_burden",
    "growth_efficiency",
    "champion_x_growth",
    "expansion_capacity",
]

CATEGORICAL_COLUMNS = ["region", "industry", "plan_tier", "segment"]
BOOLEAN_COLUMNS = ["has_exec_sponsor", "multi_product", "previous_expansion"]


def base_feature_columns() -> list[str]:
    return BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS


def engineered_feature_columns() -> list[str]:
    return base_feature_columns() + ENGINEERED_NUMERIC_COLUMNS
