# MindScope Project Overview

## Purpose
MindScope is a Python analytics prototype combining structured-data analysis, machine learning, and natural-language processing to explore workplace burnout patterns.

## Main components

### Data loading
`src/data_loader.py` loads the CSV dataset from `data/mental_health_burnout_tech_2026.csv`.

### Analytics
`src/analysis.py` provides descriptive and risk-oriented calculations for burnout, workload, sleep, and manager support.

### Correlation analysis
`src/correlation_engine.py` calculates associations between burnout and selected numeric variables. Correlation is not treated as proof of causation.

### Structured machine learning
`src/feature_engineering.py` prepares model inputs and prevents target leakage. `src/burnout_classifier.py` compares classification algorithms. `src/model_evaluation.py` provides reusable evaluation outputs.

### NLP
The NLP layer supports text preprocessing, theme detection, TF-IDF keywords, LDA topics, and burnout-language frequency analysis. Synthetic feedback is available for demonstrating the NLP pipeline.

### Application
`app.py` orchestrates the analysis. `dashboard.py` contains the Streamlit interface.

## Data flow

```text
CSV dataset
    |
    +--> Descriptive analytics
    +--> Risk analysis
    +--> Correlation analysis
    +--> Feature engineering --> ML models --> Evaluation
    +--> Feedback --> NLP --> Themes / topics / keywords
    |
    +--> Future unified dashboard and reporting
```

## Important design decision

The structured ML model and generated-feedback NLP pipeline are kept separate. Training a text classifier on feedback generated from the target variable would create a circular learning problem and could produce misleading evaluation results.

## Current status

MindScope is a portfolio/research prototype. Future work should strengthen validation, explainability, fairness analysis, privacy, security, monitoring, and dashboard/model alignment before any real-world use.
