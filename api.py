"""FastAPI service for the structured MindScope burnout classifier."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI

from src.model_service import load_or_train_model, prepare_prediction_frame

app = FastAPI(
    title="MindScope Burnout Prediction API",
    description="Structured-data burnout prediction API for portfolio/research use.",
    version="2.0.0",
)

model = load_or_train_model()


from pydantic import BaseModel


class PredictionRequest(BaseModel):
    age: float
    years_experience: float
    years_at_company: float
    salary_usd: float
    work_hours_per_week: float
    meetings_per_day: float
    team_size: float
    sleep_hours_per_night: float
    exercise_days_per_week: float
    vacation_days_taken: float
    therapy_access: int
    uses_therapy: int
    ai_tools_daily: float
    manager_support_score: float
    work_life_balance_score: float
    job_satisfaction_score: float
    social_support_score: float
    deadline_pressure_score: float
    autonomy_score: float
    stress_score: float
    phq9_score: float
    gad7_score: float
    seeks_mental_health_support: int
    job_change_intention: int
    gender: str
    country: str
    job_role: str
    seniority_level: str
    company_size: str
    industry: str
    work_mode: str


@app.get("/")
def home():
    return {
        "application": "MindScope Analytics",
        "service": "Structured Burnout Prediction API",
        "version": "2.0.0",
        "status": "Running",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "structured_burnout_classifier"}


@app.post("/predict")
def predict_burnout(request: PredictionRequest):
    frame = prepare_prediction_frame(request.model_dump())
    prediction = model.predict(frame)[0]
    probabilities = model.predict_proba(frame)[0]

    return {
        "predicted_burnout_level": prediction,
        "confidence": round(float(probabilities.max()), 4),
        "confidence_scores": {
            label: round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
        },
    }
