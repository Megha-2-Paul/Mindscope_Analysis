# MindScope Project Overview

## Purpose

MindScope is a Python analytics prototype combining structured-data analysis, machine learning, and natural-language processing to explore workplace burnout patterns.

## Main components

### Data loading
`src/data_loader.py` loads the primary dataset from `data/mental_health_burnout_tech_2026.csv`.

### Analytics
`src/analysis.py` provides descriptive and risk-oriented calculations for burnout, workload, sleep, and manager support.

### Correlation analysis
`src/correlation_engine.py` calculates correlations between burnout and selected numeric variables. The output uses non-causal language because correlation does not establish causation.

### Structured machine learning
`src/feature_engineering.py` prepares model inputs and prevents target leakage. `src/burnout_classifier.py` compares classification algorithms. `src/model_evaluation.py` provides reusable evaluation outputs.

`src/model_service.py` provides a shared model-loading/training path used by the dashboard, API, and CLI.

### NLP
The NLP layer supports text preprocessing, theme detection, TF-IDF keywords, LDA topics, and burnout-language frequency analysis. Synthetic feedback is available for demonstrating the NLP workflow.

### Applications
- `app.py` orchestrates the full analytics/NLP/ML pipeline.
- `dashboard.py` provides structured-feature Streamlit prediction and batch prediction.
- `api.py` exposes the same structured model through FastAPI.
- `predict_burnout.py` provides a command-line prediction interface.
- `train_classifier.py` is the explicit structured-model training entry point.

## Data flow

```text
CSV dataset
    |
    +--> Descriptive analytics
    +--> Risk analysis
    +--> Correlation analysis
    +--> Feature engineering --> ML models --> Evaluation
    +--> Feedback generation --> NLP --> Themes / topics / keywords
    |
    +--> Streamlit / FastAPI / CLI prediction interfaces
```

## Important design decision

The structured ML model and generated-feedback NLP pipeline are kept separate. Training a text classifier on feedback generated from burnout-related variables would create a circular learning problem and could produce misleading evaluation results.

## Current status

MindScope is a portfolio/research prototype. The structured ML pipeline, NLP exploration, dashboard, API, tests, and documentation are aligned. Future work can strengthen validation, explainability, fairness analysis, privacy, security, monitoring, and deployment hardening before any real-world use.
