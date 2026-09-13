import pandas as pd
import pytest

from src.feature_engineering import prepare_features


def make_row():
    return {
        "employee_id": 1,
        "age": 30,
        "gender": "Female",
        "country": "India",
        "job_role": "Data Scientist",
        "seniority_level": "Mid",
        "years_experience": 7,
        "years_at_company": 3.0,
        "company_size": "Mid (201-1000)",
        "industry": "Fintech",
        "work_mode": "Hybrid",
        "salary_usd": 80000,
        "work_hours_per_week": 45,
        "meetings_per_day": 4,
        "team_size": 10,
        "sleep_hours_per_night": 6.5,
        "exercise_days_per_week": 3,
        "vacation_days_taken": 12,
        "therapy_access": 1,
        "uses_therapy": 0,
        "ai_tools_daily": 1,
        "manager_support_score": 6.0,
        "work_life_balance_score": 5.5,
        "job_satisfaction_score": 6.0,
        "social_support_score": 5.0,
        "deadline_pressure_score": 6.0,
        "autonomy_score": 6.0,
        "stress_score": 6.5,
        "burnout_score": 6.0,
        "phq9_score": 5,
        "phq9_category": "Mild (5-9)",
        "gad7_score": 5,
        "gad7_category": "Mild (5-9)",
        "burnout_level": "Moderate",
        "seeks_mental_health_support": 0,
        "job_change_intention": 0,
    }


def test_burnout_score_is_not_used_as_a_feature():
    X, y = prepare_features(pd.DataFrame([make_row()]))
    assert "burnout_score" not in X.columns
    assert y.iloc[0] == "Moderate"


def test_missing_required_column_is_rejected():
    df = pd.DataFrame([make_row()]).drop(columns=["stress_score"])
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_features(df)
