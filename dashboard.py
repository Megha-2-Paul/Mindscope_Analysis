import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MindScope Analytics",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/burnout_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# ==========================================
# INTERPRETATION FUNCTION
# ==========================================

def get_interpretation(level):

    if level == "Low":

        return (
            "Healthy engagement and low burnout risk."
        )

    elif level == "Moderate":

        return (
            "Some signs of stress detected. Monitor employee wellbeing."
        )

    elif level == "High":

        return (
            "Elevated burnout risk. Consider workload adjustments."
        )

    else:

        return (
            "Critical burnout indicators detected. Immediate intervention recommended."
        )

# ==========================================
# TITLE
# ==========================================

st.title(
    "🧠 MindScope Burnout Analytics"
)

st.markdown(
    """
    AI-Powered Employee Burnout Prediction using
    NLP, Machine Learning, FastAPI and Streamlit.
    """
)

st.divider()

# ==========================================
# SAMPLE FEEDBACKS
# ==========================================

st.subheader(
    "Try Sample Feedback"
)

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🟢 Low Burnout Example"
    ):

        st.session_state.feedback = (
            "My manager is supportive and I enjoy working with my team. "
            "The organization provides excellent growth opportunities."
        )

with col2:

    if st.button(
        "🔴 Severe Burnout Example"
    ):

        st.session_state.feedback = (
            "The workload has become overwhelming and I rarely feel rested. "
            "The pressure to meet deadlines has become mentally exhausting."
        )

# ==========================================
# SINGLE PREDICTION
# ==========================================

st.header(
    "👤 Individual Burnout Prediction"
)

feedback = st.text_area(
    "Employee Feedback",
    value=st.session_state.get(
        "feedback",
        ""
    ),
    height=200
)

if st.button(
    "🚀 Predict Burnout"
):

    if feedback.strip() == "":

        st.warning(
            "Please enter employee feedback."
        )

    else:

        feedback_vector = (
            vectorizer.transform(
                [feedback]
            )
        )

        prediction = model.predict(
            feedback_vector
        )[0]

        probabilities = (
            model.predict_proba(
                feedback_vector
            )[0]
        )

        confidence = max(
            probabilities
        )

        st.subheader(
            "Prediction Result"
        )

        if prediction == "Low":

            st.success(
                f"Predicted Burnout Level: {prediction}"
            )

        elif prediction == "Moderate":

            st.info(
                f"Predicted Burnout Level: {prediction}"
            )

        elif prediction == "High":

            st.warning(
                f"Predicted Burnout Level: {prediction}"
            )

        else:

            st.error(
                f"Predicted Burnout Level: {prediction}"
            )

        st.metric(
            "Model Confidence",
            f"{confidence:.2%}"
        )

        confidence_df = pd.DataFrame(
            {
                "Burnout Level": model.classes_,
                "Probability": probabilities
            }
        )

        st.subheader(
            "Confidence Scores"
        )

        st.dataframe(
            confidence_df,
            use_container_width=True
        )

        st.subheader(
            "Prediction Probability Distribution"
        )

        st.bar_chart(
            confidence_df.set_index(
                "Burnout Level"
            )
        )

        st.subheader(
            "Interpretation"
        )

        interpretation = get_interpretation(
            prediction
        )

        if prediction == "Low":

            st.success(
                interpretation
            )

        elif prediction == "Moderate":

            st.info(
                interpretation
            )

        elif prediction == "High":

            st.warning(
                interpretation
            )

        else:

            st.error(
                interpretation
            )

# ==========================================
# BATCH PREDICTION
# ==========================================

st.divider()

st.header(
    "📂 Batch Prediction"
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(
        uploaded_file
    )

    st.subheader(
        "Uploaded Data Preview"
    )

    st.dataframe(
        batch_df.head(),
        use_container_width=True
    )

    if "employee_feedback" not in batch_df.columns:

        st.error(
            "CSV must contain an 'employee_feedback' column."
        )

    else:

        if st.button(
            "📊 Run Batch Prediction"
        ):

            feedback_vectors = (
                vectorizer.transform(
                    batch_df[
                        "employee_feedback"
                    ]
                )
            )

            predictions = (
                model.predict(
                    feedback_vectors
                )
            )

            probabilities = (
                model.predict_proba(
                    feedback_vectors
                )
            )

            confidence_scores = (
                probabilities.max(axis=1)
            )

            batch_df[
                "burnout_prediction"
            ] = predictions

            batch_df[
                "confidence"
            ] = (
                confidence_scores * 100
            ).round(2)

            batch_df[
                "interpretation"
            ] = (
                batch_df[
                    "burnout_prediction"
                ].apply(
                    get_interpretation
                )
            )

            class_labels = list(
                model.classes_
            )

            batch_df["Low_Prob"] = (
                probabilities[
                    :,
                    class_labels.index("Low")
                ] * 100
            ).round(2)

            batch_df["Moderate_Prob"] = (
                probabilities[
                    :,
                    class_labels.index("Moderate")
                ] * 100
            ).round(2)

            batch_df["High_Prob"] = (
                probabilities[
                    :,
                    class_labels.index("High")
                ] * 100
            ).round(2)

            batch_df["Severe_Prob"] = (
                probabilities[
                    :,
                    class_labels.index("Severe")
                ] * 100
            ).round(2)

            st.success(
                "Batch prediction completed successfully."
            )

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                batch_df,
                use_container_width=True
            )

            # Summary Metrics

            st.subheader(
                "Burnout Distribution"
            )

            burnout_counts = (
                batch_df[
                    "burnout_prediction"
                ]
                .value_counts()
            )

            st.bar_chart(
                burnout_counts
            )

            # Download

            csv = batch_df.to_csv(
                index=False
            )

            st.download_button(
                label="⬇ Download Results",
                data=csv,
                file_name="burnout_predictions.csv",
                mime="text/csv"
            )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "MindScope Analytics • NLP + Machine Learning + FastAPI + Streamlit"
)