import pandas as pd

from src.burnout_classifier import (
    train_burnout_classifier
)

print(
    "\nLoading NLP dataset..."
)

df = pd.read_csv(
    "data/nlp_employee_feedback.csv"
)

print(
    f"Dataset Size: {len(df):,}"
)

accuracy, report = (
    train_burnout_classifier(df)
)

print(
    "\n========== BURNOUT CLASSIFIER ==========\n"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    "\nClassification Report:\n"
)

print(report)

print(
    "\nModel saved successfully."
)