"""Shared model-loading utilities for the structured burnout classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.burnout_classifier import train_burnout_classifier

DATA_PATH = Path("data/mental_health_burnout_tech_2026.csv")
MODEL_PATH = Path("models/burnout_structured_model.pkl")
DEFAULT_SAMPLE_SIZE = 8000


def load_or_train_model(sample_size: int = DEFAULT_SAMPLE_SIZE) -> Pipeline:
    """Load the structured model, training it from the project dataset if needed."""
    if MODEL_PATH.exists():
        import joblib
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    model, _, _, _, _ = train_burnout_classifier(df)
    return model


def prepare_prediction_frame(payload: dict) -> pd.DataFrame:
    """Convert a validated prediction payload into a one-row DataFrame."""
    return pd.DataFrame([payload])
