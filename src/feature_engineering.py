"""Feature preparation for the structured burnout prediction model."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


TARGET_COLUMN = "burnout_level"

# burnout_score is intentionally excluded because burnout_level is derived from
# the underlying burnout measurement. Including it would create target leakage.
EXCLUDED_COLUMNS = {
    "employee_id",
    "burnout_score",
    "burnout_level",
    "phq9_category",
    "gad7_category",
}

NUMERIC_FEATURES = [
    "age",
    "years_experience",
    "years_at_company",
    "salary_usd",
    "work_hours_per_week",
    "meetings_per_day",
    "team_size",
    "sleep_hours_per_night",
    "exercise_days_per_week",
    "vacation_days_taken",
    "therapy_access",
    "uses_therapy",
    "ai_tools_daily",
    "manager_support_score",
    "work_life_balance_score",
    "job_satisfaction_score",
    "social_support_score",
    "deadline_pressure_score",
    "autonomy_score",
    "stress_score",
    "phq9_score",
    "gad7_score",
    "seeks_mental_health_support",
    "job_change_intention",
]

CATEGORICAL_FEATURES = [
    "gender",
    "country",
    "job_role",
    "seniority_level",
    "company_size",
    "industry",
    "work_mode",
]


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate the dataset and return leakage-safe features and target."""

    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")

    return X, y
