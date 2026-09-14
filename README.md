# MindScope — Employee Burnout Intelligence

MindScope is a Python-based analytics and machine-learning project for exploring workplace burnout patterns, identifying associated risk factors, comparing predictive models, and extracting themes from employee feedback.

> **Project status:** Portfolio / research prototype. It is not a clinical diagnostic system and should not be used as the sole basis for employment, HR, or medical decisions.

## What MindScope does

MindScope combines two complementary analysis tracks:

1. **Structured-data analytics and ML** — analyzes workplace and wellbeing variables and predicts the dataset's `burnout_level` category.
2. **NLP analysis** — processes employee feedback to identify themes, keywords, topics, and burnout-related language. The current project can generate synthetic feedback for demonstration purposes; this text is intentionally kept separate from the primary structured ML training pipeline.

## Key capabilities

- Burnout and workforce descriptive analytics
- Risk-oriented analysis of workload, sleep, stress, and manager support
- Correlation analysis across workplace and wellbeing variables
- Feature engineering with explicit target-leakage prevention
- Comparison of Logistic Regression, Random Forest, and Gradient Boosting
- Cross-validation and classification metrics
- Confusion-matrix evaluation
- TF-IDF keyword extraction
- LDA topic modelling
- Theme detection and burnout-language analysis
- Streamlit dashboard and batch-prediction components

## Architecture

```text
                    MindScope
                       |
          +------------+------------+
          |                         |
          v                         v
   Structured Data             Feedback Text
          |                         |
          v                         v
 Feature Engineering          Preprocessing
          |                         |
          v                         v
  ML Model Comparison       NLP / Theme Analysis
          |                         |
          v                         v
     Risk Prediction        Topics / Keywords
          |                         |
          +------------+------------+
                       |
                       v
                  Insights
```

## ML pipeline

The primary burnout classifier uses structured employee features rather than the generated feedback text. `burnout_score` is excluded from model features because it is directly related to the target and could create target leakage.

The current model candidates are:

- Logistic Regression — interpretable baseline
- Random Forest — nonlinear tree-based model
- Gradient Boosting — nonlinear boosting model

Models are compared using cross-validation and weighted F1, with additional holdout evaluation using accuracy, precision, recall, F1, classification report, and confusion matrix.

## NLP pipeline

The NLP component currently supports:

```text
Employee feedback
       |
       v
Text preprocessing
       |
       +----> Theme detection
       |
       +----> TF-IDF keywords
       |
       +----> LDA topics
       |
       +----> Burnout-language frequency
```

Generated feedback is useful for demonstrating the NLP pipeline, but it is not treated as independent real-world employee testimony.

## Dataset

The project currently uses `data/mental_health_burnout_tech_2026.csv`. The dataset contains employee/workplace characteristics, wellbeing measures, and burnout labels. See [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for the documented fields.

## Project structure

```text
Mindscope_Analysis/
├── app.py
├── dashboard.py
├── requirements.txt
├── data/
│   ├── mental_health_burnout_tech_2026.csv
│   └── ...
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── analysis.py
│   ├── burnout_classifier.py
│   ├── correlation_engine.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── nlp_engine.py
│   ├── text_generator.py
│   └── ...
├── tests/
└── docs/
    ├── PROJECT_OVERVIEW.md
    ├── ML_METHODOLOGY.md
    ├── DATA_DICTIONARY.md
    └── NLP_PIPELINE.md
```

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Megha-2-Paul/Mindscope_Analysis.git
cd Mindscope_Analysis
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the analysis

Run the main analytics pipeline with:

```bash
python app.py
```

The application loads the dataset, performs the descriptive/risk analyses, runs the NLP components that are enabled, and trains/evaluates the structured burnout classifier.

## Running tests

Run the test suite with:

```bash
pytest
```

## Dashboard

The repository also contains a Streamlit dashboard. Run it with:

```bash
streamlit run dashboard.py
```

The dashboard should be considered a prototype interface until its prediction inputs are fully aligned with the structured ML model and production data contract.

## Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) — system architecture and components
- [ML Methodology](docs/ML_METHODOLOGY.md) — feature design, leakage prevention, models, and evaluation
- [Data Dictionary](docs/DATA_DICTIONARY.md) — dataset fields and modeling roles
- [NLP Pipeline](docs/NLP_PIPELINE.md) — feedback generation, preprocessing, themes, keywords, and topics
- [Contributing](CONTRIBUTING.md) — development and contribution guidance

## Limitations and responsible use

MindScope is a portfolio/research prototype. Results depend on the quality, representativeness, and construction of the underlying dataset. Associations should not automatically be interpreted as causal relationships. Burnout and mental-health-related measurements are sensitive and should be handled with appropriate privacy, security, consent, and governance controls.

The model should not be used to diagnose a medical condition, make automated employment decisions, or determine an employee's fitness for work.

## Roadmap

Planned improvements include:

- Explainable AI using SHAP or comparable techniques
- Stronger statistical inference and confidence intervals
- A fully aligned interactive dashboard
- Individual risk profiles and contributing-factor explanations
- Model fairness/performance analysis across relevant groups
- Better data validation and monitoring
- CI-based automated testing
- Deployment-ready API and application architecture

## Technology

Python, Pandas, NumPy, scikit-learn, Matplotlib/Plotly, NLP tooling, Joblib, Streamlit, and pytest.

## Author

**Megha Paul**

GitHub: https://github.com/Megha-2-Paul
