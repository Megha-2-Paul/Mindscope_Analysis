import pandas as pd

from src.model_service import prepare_prediction_frame


def test_prediction_frame_preserves_feature_values():
    payload = {
        "age": 30,
        "years_experience": 5,
        "years_at_company": 2,
        "salary_usd": 70000,
        "work_hours_per_week": 45,
        "meetings_per_day": 4,
        "team_size": 10,
        "sleep_hours_per_night": 7,
        "exercise_days_per_week": 3,
        "vacation_days_taken": 12,
        "therapy_access": 0,
        "uses_therapy": 0,
        "ai_tools_daily": 1,
        "manager_support_score": 6,
        "work_life_balance_score": 6,
        "job_satisfaction_score": 6,
        "social_support_score": 6,
        "deadline_pressure_score": 5,
        "autonomy_score": 6,
        "stress_score": 5,
        "phq9_score": 5,
        "gad7_score": 5,
        "seeks_mental_health_support": 0,
        "job_change_intention": 0,
        "gender": "Female",
        "country": "India",
        "job_role": "Data Scientist",
        "seniority_level": "Mid",
        "company_size": "Mid (201-1000)",
        "industry": "Technology",
        "work_mode": "Hybrid",
    }

    frame = prepare_prediction_frame(payload)

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (1, len(payload))
    assert frame.loc[0, "job_role"] == "Data Scientist"
