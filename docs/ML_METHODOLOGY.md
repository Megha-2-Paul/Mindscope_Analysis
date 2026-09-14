# ML Methodology

## Objective

The structured-data classifier predicts the dataset's `burnout_level` category from available employee/workplace features.

## Target

`burnout_level` is the classification target.

## Leakage prevention

`burnout_score` is excluded from model features because it is directly related to the target label. Using it would allow the model to learn information that would not be available in a legitimate prediction setting.

Generated employee feedback is also not used as the primary structured ML training input because the current feedback generator creates text using burnout-related values.

## Candidate models

The current pipeline compares:

1. Logistic Regression — interpretable baseline.
2. Random Forest — nonlinear tree ensemble.
3. Gradient Boosting — nonlinear boosting model.

## Validation

The data is split into training and holdout sets using stratification so that class proportions are preserved. Candidate models are evaluated with cross-validation on the training data.

Weighted F1 is used as the primary model-selection metric because the target contains multiple classes and the project should balance precision and recall rather than optimize accuracy alone.

## Reported metrics

The evaluation layer reports:

- Accuracy
- Precision
- Recall
- Weighted F1
- Classification report
- Confusion matrix
- Cross-validation performance

## Interpretation

Model performance should be interpreted in the context of the dataset. A high score on a synthetic or constructed dataset does not establish real-world predictive validity.

The project uses predictive modelling for analytical demonstration. It is not intended to diagnose conditions or automate consequential decisions about people.

## Future ML improvements

- Hyperparameter tuning
- Calibration analysis
- ROC-AUC and precision-recall curves where appropriate
- SHAP-based explainability
- Feature-importance stability analysis
- Group-wise performance and fairness checks
- External validation on an independent dataset
- Model and data drift monitoring
