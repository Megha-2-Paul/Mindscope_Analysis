from fastapi import FastAPI
from pydantic import BaseModel

import joblib

# ==========================================
# LOAD TRAINED MODEL & VECTORIZER
# ==========================================

model = joblib.load(
    "models/burnout_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="MindScope Burnout Prediction API",
    description="AI-powered employee burnout prediction using NLP and Machine Learning",
    version="1.0.0"
)

# ==========================================
# REQUEST MODEL
# ==========================================

class FeedbackRequest(BaseModel):
    feedback: str


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "application": "MindScope Analytics",
        "service": "Burnout Prediction API",
        "version": "1.0.0",
        "status": "Running"
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict_burnout(
    request: FeedbackRequest
):

    # Convert text to TF-IDF features
    feedback_vector = vectorizer.transform(
        [request.feedback]
    )

    # Predict burnout level
    prediction = model.predict(
        feedback_vector
    )[0]

    # Predict probabilities
    probabilities = model.predict_proba(
        feedback_vector
    )[0]

    # Highest confidence
    confidence = max(
        probabilities
    )

    # Store all confidence scores
    confidence_scores = {}

    for label, prob in zip(
        model.classes_,
        probabilities
    ):

        confidence_scores[label] = round(
            float(prob),
            4
        )

    return {

        "feedback":
            request.feedback,

        "predicted_burnout_level":
            prediction,

        "confidence":
            round(
                float(confidence),
                4
            ),

        "confidence_scores":
            confidence_scores
    }