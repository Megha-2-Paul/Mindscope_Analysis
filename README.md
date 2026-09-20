# MindScope — Employee Burnout Intelligence

MindScope is a Python portfolio/research project that combines structured-data analytics, machine learning, and NLP exploration to study workplace burnout patterns.

> **Status:** Portfolio / research prototype. It is not a clinical diagnostic system and should not be used as the sole basis for employment, HR, or medical decisions.

## What MindScope does

MindScope has two intentionally separate analysis tracks:

1. **Structured ML:** predicts the dataset's `burnout_level` category from employee/workplace features.
2. **NLP exploration:** analyzes employee feedback for themes, keywords, topics, and burnout-related language. The current feedback generator is synthetic and is used only to demonstrate the NLP workflow.

The structured ML model does **not** use generated feedback.

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
          v                         +----> Themes
     ML Comparison                    +----> TF-IDF Keywords
          |                           +----> LDA Topics
          v                           +----> Burnout Language
   Best Classifier
          |
          v
    Burnout Level
```

## Structured ML pipeline

The target is `burnout_level`.

`burnout_score` is deliberately excluded because it is directly related to the target and would create target leakage. `phq9_category` and `gad7_category` are also excluded because their numeric counterparts are already represented and the categorical versions are sensitive/redundant.

Candidate models:

- Logistic Regression
- Random Forest
- Gradient Boosting

Model selection uses 5-fold cross-validation with weighted F1. The selected model is then evaluated on a stratified holdout set using accuracy, precision, recall, F1, classification report, and confusion matrix.

## NLP pipeline

The NLP track supports:

- Text preprocessing
- Rule-based theme detection
- TF-IDF keyword extraction
- LDA topic modelling
- Burnout-language frequency analysis

Synthetic feedback is generated from structured attributes for demonstration only. Because that feedback is constructed from burnout-related variables, it is **not** treated as independent employee testimony and is not used as the primary structured ML training input.

## Streamlit dashboard

Run:

```bash
streamlit run dashboard.py
```

The dashboard uses the same structured ML pipeline as the training code. If a trained model artifact is not present locally, it trains the model from the project dataset and caches it for the session.

The dashboard supports:

- Individual structured-feature prediction
- Class probabilities
- Batch CSV prediction
- Downloadable prediction results

## FastAPI API

Run:

```bash
uvicorn api:app --reload
```

Endpoints:

- `GET /` — service information
- `GET /health` — health check
- `POST /predict` — structured burnout prediction

The request schema mirrors the model's required features. No employee feedback text is required for prediction.

## Installation

```bash
git clone https://github.com/Megha-2-Paul/Mindscope_Analysis.git
cd Mindscope_Analysis
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the automated tests:

```bash
pytest
```

Train the structured model explicitly:

```bash
python train_classifier.py
```

Run the full analytics/NLP pipeline:

```bash
python app.py
```

## Project structure

```text
Mindscope_Analysis/
├── app.py
├── api.py
├── dashboard.py
├── predict_burnout.py
├── train_classifier.py
├── data/
├── docs/
├── models/                 # local generated model artifacts
├── notebooks/
├── reports/
├── src/
│   ├── analysis.py
│   ├── burnout_classifier.py
│   ├── correlation_engine.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── model_service.py
│   ├── nlp_engine.py
│   ├── text_generator.py
│   └── ...
├── tests/
├── visualizations/
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Responsible use and limitations

- Correlation is not evidence of causation.
- Dataset quality and representativeness determine how meaningful the results are.
- Mental-health-related fields are sensitive and require appropriate privacy, consent, and governance.
- Synthetic feedback is not real employee testimony.
- The model has not been externally validated for real-world workplace use.
- Predictions should not be used to diagnose medical conditions or make automated employment decisions.

## Future improvements

Potential future work includes:

- SHAP or comparable explainability
- Model calibration
- External validation
- Group-wise performance/fairness analysis
- Data validation and monitoring
- CI-based testing
- Deployment hardening

These are optional future improvements, not required for the current portfolio prototype.

## Author

**Megha Paul**

GitHub: https://github.com/Megha-2-Paul
