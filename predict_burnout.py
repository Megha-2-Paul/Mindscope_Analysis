"""Command-line prediction for the structured MindScope model."""

from src.model_service import load_or_train_model, prepare_prediction_frame

model = load_or_train_model()

print("\nMindScope Structured Burnout Predictor\n")
print("Enter employee/workplace features as requested below.\n")

payload = {
    "age": float(input("Age: ")),
    "years_experience": float(input("Years of experience: ")),
    "years_at_company": float(input("Years at company: ")),
    "salary_usd": float(input("Salary (USD): ")),
    "work_hours_per_week": float(input("Work hours/week: ")),
    "meetings_per_day": float(input("Meetings/day: ")),
    "team_size": float(input("Team size: ")),
    "sleep_hours_per_night": float(input("Sleep hours/night: ")),
    "exercise_days_per_week": float(input("Exercise days/week: ")),
    "vacation_days_taken": float(input("Vacation days taken: ")),
    "therapy_access": int(input("Therapy access (0/1): ")),
    "uses_therapy": int(input("Uses therapy (0/1): ")),
    "ai_tools_daily": float(input("AI tools daily: ")),
    "manager_support_score": float(input("Manager support score: ")),
    "work_life_balance_score": float(input("Work-life balance score: ")),
    "job_satisfaction_score": float(input("Job satisfaction score: ")),
    "social_support_score": float(input("Social support score: ")),
    "deadline_pressure_score": float(input("Deadline pressure score: ")),
    "autonomy_score": float(input("Autonomy score: ")),
    "stress_score": float(input("Stress score: ")),
    "phq9_score": float(input("PHQ-9 score: ")),
    "gad7_score": float(input("GAD-7 score: ")),
    "seeks_mental_health_support": int(input("Seeks mental-health support (0/1): ")),
    "job_change_intention": int(input("Job-change intention (0/1): ")),
    "gender": input("Gender: "),
    "country": input("Country: "),
    "job_role": input("Job role: "),
    "seniority_level": input("Seniority level: "),
    "company_size": input("Company size: "),
    "industry": input("Industry: "),
    "work_mode": input("Work mode: "),
}

frame = prepare_prediction_frame(payload)
prediction = model.predict(frame)[0]
probabilities = model.predict_proba(frame)[0]

print(f"\nPredicted Burnout Level: {prediction}\n")
for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability:.2%}")
