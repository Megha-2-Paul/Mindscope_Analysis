import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.svm import (
    LinearSVC
)

from sklearn.naive_bayes import (
    MultinomialNB
)

from sklearn.ensemble import (
    RandomForestClassifier
)

print("\nLoading dataset...\n")

df = pd.read_csv(
    "data/nlp_employee_feedback.csv"
)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df["employee_feedback"]

y = df["burnout_level"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

# import pandas as pd

# df = pd.read_csv(
#     "data/nlp_employee_feedback.csv"
# )

# print(
#     df["burnout_level"].value_counts()
# )

# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

# ==========================================
# MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Linear SVC":
        LinearSVC(),

    "Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

results = []

# ==========================================
# TRAIN & EVALUATE
# ==========================================

for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append(
        [name, accuracy]
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

# ==========================================
# RESULTS
# ==========================================

print(
    "\n========== MODEL COMPARISON ==========\n"
)

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy"
    ]
)

print(
    results_df.sort_values(
        by="Accuracy",
        ascending=False
    )
)