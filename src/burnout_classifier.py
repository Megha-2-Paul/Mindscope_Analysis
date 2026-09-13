"""Structured-data burnout classification pipeline."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_features
from src.model_evaluation import cross_validate_models, evaluate_classifier


MODEL_PATH = "models/burnout_structured_model.pkl"


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-safe preprocessing for numeric and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> Dict[str, object]:
    """Return the candidate classifiers used for model comparison."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def _build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def train_burnout_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    cv: int = 5,
) -> Tuple[Pipeline, ColumnTransformer, float, str, pd.DataFrame]:
    """Train, compare and save the best structured-data burnout classifier.

    The target is ``burnout_level``. ``burnout_score`` is deliberately excluded
    because it is a direct burnout measurement and would leak target information.
    """

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    candidate_pipelines = {
        name: _build_pipeline(classifier)
        for name, classifier in build_models().items()
    }

    comparison = cross_validate_models(
        candidate_pipelines,
        X_train,
        y_train,
        cv=cv,
    )

    best_name = comparison.iloc[0]["model"]
    best_pipeline = candidate_pipelines[best_name]
    best_pipeline.fit(X_train, y_train)

    metrics, report, _ = evaluate_classifier(
        best_pipeline,
        X_test,
        y_test,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    return (
        best_pipeline,
        best_pipeline.named_steps["preprocessor"],
        metrics["accuracy"],
        report,
        comparison,
    )
