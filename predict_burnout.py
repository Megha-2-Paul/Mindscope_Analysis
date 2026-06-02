import joblib

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/burnout_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("\n====================================")
print(" MINDSCOPE BURNOUT PREDICTOR ")
print("====================================\n")

# ==========================================
# USER INPUT
# ==========================================

feedback = input(
    "Enter employee feedback:\n\n"
)

# ==========================================
# TRANSFORM TEXT
# ==========================================

feedback_vector = vectorizer.transform(
    [feedback]
)

# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(
    feedback_vector
)[0]

probabilities = model.predict_proba(
    feedback_vector
)[0]

# ==========================================
# OUTPUT
# ==========================================

print("\n========== PREDICTION ==========\n")

print(
    f"Predicted Burnout Level: {prediction}"
)

print("\nConfidence Scores:\n")

for label, prob in zip(
    model.classes_,
    probabilities
):

    print(
        f"{label}: {prob:.2%}"
    )