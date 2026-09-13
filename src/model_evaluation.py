"""Reusable evaluation helpers for burnout classification models."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def cross_validate_models(models, X, y, cv=5) -> pd.DataFrame:
    """Compare candidate classifiers using weighted F1 cross-validation."""

    rows = []
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "cv_f1_mean": scores.mean(),
                "cv_f1_std": scores.std(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("cv_f1_mean", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_classifier(model, X_test, y_test) -> Tuple[Dict, str, object]:
    """Evaluate a fitted classifier on the held-out test set."""

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }

    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)

    return metrics, report, matrix
