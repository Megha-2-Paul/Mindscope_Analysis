"""Train the structured MindScope burnout classifier."""

import pandas as pd

from src.burnout_classifier import train_burnout_classifier

DATA_PATH = "data/mental_health_burnout_tech_2026.csv"

df = pd.read_csv(DATA_PATH)

if len(df) > 8000:
    df = df.sample(n=8000, random_state=42)

model, _, accuracy, report, comparison = train_burnout_classifier(df)

print("\n========== MODEL COMPARISON ==========\n")
print(comparison.to_string(index=False))
print(f"\nHoldout accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(report)
print("\nStructured model saved to models/burnout_structured_model.pkl")
