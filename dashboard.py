"""MindScope Streamlit dashboard for structured burnout prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.model_service import load_or_train_model, prepare_prediction_frame

st.set_page_config(page_title="MindScope Analytics", page_icon="🧠", layout="wide")
st.title("🧠 MindScope Burnout Analytics")
st.caption(
    "Structured-data machine learning for portfolio/research use. "
    "This prototype is not a clinical diagnostic or employment decision system."
)

@st.cache_resource
def get_model():
    return load_or_train_model()

model = get_model()

st.info(
    "This dashboard uses the same structured-data pipeline described in the "
    "repository documentation. The model does not use generated employee feedback."
)

tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])

with tab1:
    st.header("Individual Burnout Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 80, 30)
        years_experience = st.number_input("Years of Experience", 0.0, 50.0, 5.0)
        years_at_company = st.number_input("Years at Company", 0.0, 50.0, 2.0)
        salary_usd = st.number_input("Salary (USD)", 0.0, 2000000.0, 70000.0)
        work_hours_per_week = st.number_input("Work Hours / Week", 1.0, 100.0, 45.0)
        meetings_per_day = st.number_input("Meetings / Day", 0.0, 20.0, 4.0)
        team_size = st.number_input("Team Size", 1.0, 1000.0, 10.0)
        sleep_hours_per_night = st.number_input("Sleep Hours / Night", 0.0, 16.0, 7.0)

    with c2:
        exercise_days_per_week = st.number_input("Exercise Days / Week", 0.0, 7.0, 3.0)
        vacation_days_taken = st.number_input("Vacation Days Taken", 0.0, 100.0, 12.0)
        therapy_access = st.selectbox("Therapy Access", [0, 1], format_func=lambda x: "Yes" if x else "No")
        uses_therapy = st.selectbox("Uses Therapy", [0, 1], format_func=lambda x: "Yes" if x else "No")
        ai_tools_daily = st.number_input("AI Tools Daily", 0.0, 20.0, 1.0)
        manager_support_score = st.slider("Manager Support", 0.0, 10.0, 6.0)
        work_life_balance_score = st.slider("Work-Life Balance", 0.0, 10.0, 6.0)
        job_satisfaction_score = st.slider("Job Satisfaction", 0.0, 10.0, 6.0)

    with c3:
        social_support_score = st.slider("Social Support", 0.0, 10.0, 6.0)
        deadline_pressure_score = st.slider("Deadline Pressure", 0.0, 10.0, 5.0)
        autonomy_score = st.slider("Autonomy", 0.0, 10.0, 6.0)
        stress_score = st.slider("Stress", 0.0, 10.0, 5.0)
        phq9_score = st.number_input("PHQ-9 Score", 0.0, 27.0, 5.0)
        gad7_score = st.number_input("GAD-7 Score", 0.0, 21.0, 5.0)
        seeks_mental_health_support = st.selectbox("Seeks Mental-Health Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
        job_change_intention = st.selectbox("Job-Change Intention", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.subheader("Profile")
    p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
    with p1:
        gender = st.selectbox("Gender", ["Female", "Male", "Non-binary", "Other"])
    with p2:
        country = st.text_input("Country", "India")
    with p3:
        job_role = st.text_input("Job Role", "Data Scientist")
    with p4:
        seniority_level = st.text_input("Seniority", "Mid")
    with p5:
        company_size = st.text_input("Company Size", "Mid (201-1000)")
    with p6:
        industry = st.text_input("Industry", "Technology")
    with p7:
        work_mode = st.selectbox("Work Mode", ["Remote", "Hybrid", "On-site"])

    if st.button("🚀 Predict Burnout", type="primary"):
        payload = {
            "age": age, "years_experience": years_experience, "years_at_company": years_at_company,
            "salary_usd": salary_usd, "work_hours_per_week": work_hours_per_week,
            "meetings_per_day": meetings_per_day, "team_size": team_size,
            "sleep_hours_per_night": sleep_hours_per_night, "exercise_days_per_week": exercise_days_per_week,
            "vacation_days_taken": vacation_days_taken, "therapy_access": therapy_access,
            "uses_therapy": uses_therapy, "ai_tools_daily": ai_tools_daily,
            "manager_support_score": manager_support_score, "work_life_balance_score": work_life_balance_score,
            "job_satisfaction_score": job_satisfaction_score, "social_support_score": social_support_score,
            "deadline_pressure_score": deadline_pressure_score, "autonomy_score": autonomy_score,
            "stress_score": stress_score, "phq9_score": phq9_score, "gad7_score": gad7_score,
            "seeks_mental_health_support": seeks_mental_health_support, "job_change_intention": job_change_intention,
            "gender": gender, "country": country, "job_role": job_role, "seniority_level": seniority_level,
            "company_size": company_size, "industry": industry, "work_mode": work_mode,
        }
        frame = prepare_prediction_frame(payload)
        prediction = model.predict(frame)[0]
        probabilities = model.predict_proba(frame)[0]
        st.subheader("Prediction Result")
        st.success(f"Predicted Burnout Level: **{prediction}**")
        result = pd.DataFrame({"Burnout Level": model.classes_, "Probability": probabilities})
        st.dataframe(result, use_container_width=True)
        st.bar_chart(result.set_index("Burnout Level"))

with tab2:
    st.header("Batch Prediction")
    st.write(
        "Upload a CSV containing all model input columns listed in "
        "docs/DATA_DICTIONARY.md, excluding employee_id, burnout_score, burnout_level, "
        "phq9_category and gad7_category."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
        missing = sorted(set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(batch.columns))
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            st.dataframe(batch.head(), use_container_width=True)
            if st.button("📊 Run Batch Prediction", type="primary"):
                probabilities = model.predict_proba(batch)
                batch["burnout_prediction"] = model.predict(batch)
                batch["confidence"] = probabilities.max(axis=1)
                st.success("Batch prediction completed.")
                st.dataframe(batch, use_container_width=True)
                st.download_button(
                    "⬇ Download Results", batch.to_csv(index=False),
                    "burnout_predictions.csv", "text/csv"
                )

st.divider()
st.caption("MindScope Analytics • Structured ML + NLP exploration • Portfolio/research prototype")
