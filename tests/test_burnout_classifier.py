import pandas as pd

from src.burnout_classifier import train_burnout_classifier


def make_dataset(rows_per_class=6):
    rows = []
    levels = ["Low", "Moderate", "High", "Severe"]

    for class_index, level in enumerate(levels):
        for i in range(rows_per_class):
            rows.append(
                {
                    "employee_id": class_index * rows_per_class + i,
                    "age": 25 + i,
                    "gender": "Female" if i % 2 else "Male",
                    "country": "India" if i % 2 else "USA",
                    "job_role": "Data Scientist" if i % 2 else "Software Engineer",
                    "seniority_level": "Mid",
                    "years_experience": 3 + i,
                    "years_at_company": 1.0 + i / 2,
                    "company_size": "Mid (201-1000)",
                    "industry": "Fintech",
                    "work_mode": "Hybrid",
                    "salary_usd": 70000 + i * 1000,
                    "work_hours_per_week": 40 + class_index * 5,
                    "meetings_per_day": 3 + class_index,
                    "team_size": 8 + i,
                    "sleep_hours_per_night": 7 - class_index * 0.7,
                    "exercise_days_per_week": 4 - min(class_index, 3),
                    "vacation_days_taken": 15 - class_index * 2,
                    "therapy_access": i % 2,
                    "uses_therapy": 0,
                    "ai_tools_daily": 1,
                    "manager_support_score": 8 - class_index,
                    "work_life_balance_score": 8 - class_index,
                    "job_satisfaction_score": 8 - class_index,
                    "social_support_score": 7 - class_index * 0.5,
                    "deadline_pressure_score": 2 + class_index * 2,
                    "autonomy_score": 8 - class_index,
                    "stress_score": 2 + class_index * 2,
                    "burnout_score": 2 + class_index * 2,
                    "phq9_score": class_index * 4,
                    "phq9_category": "None (0-4)",
                    "gad7_score": class_index * 3,
                    "gad7_category": "None (0-4)",
                    "burnout_level": level,
                    "seeks_mental_health_support": class_index > 1,
                    "job_change_intention": class_index > 1,
                }
            )

    return pd.DataFrame(rows)


def test_classifier_returns_model_and_comparison(tmp_path, monkeypatch):
    import src.burnout_classifier as classifier

    monkeypatch.setattr(classifier, "MODEL_PATH", str(tmp_path / "model.pkl"))

    model, preprocessor, accuracy, report, comparison = train_burnout_classifier(
        make_dataset(), cv=3
    )

    assert hasattr(model, "predict")
    assert preprocessor is not None
    assert 0 <= accuracy <= 1
    assert "precision" in report.lower()
    assert len(comparison) == 3
    assert comparison.iloc[0]["cv_f1_mean"] >= comparison.iloc[-1]["cv_f1_mean"]
