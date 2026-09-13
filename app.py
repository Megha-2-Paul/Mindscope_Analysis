"""MindScope Analytics command-line application."""

import random
import time

from src.analysis import (
    calculate_average_burnout,
    highest_burnout_role,
    sleep_depression_analysis,
    high_workload_risk,
    sleep_risk_analysis,
    manager_support_risk,
)
from src.burnout_classifier import train_burnout_classifier
from src.correlation_engine import burnout_correlations
from src.data_loader import load_data
from src.insight_generator import generate_insights, generate_risk_insights
from src.keyword_extractor import extract_top_keywords
from src.nlp_burnout_analysis import burnout_word_frequency
from src.nlp_engine import detect_themes
from src.nlp_insights import generate_theme_insights
from src.text_generator import generate_feedback
from src.text_preprocessor import preprocess_text
from src.topic_modeler import extract_topics


DEVELOPMENT_MODE = True
RUN_TOPIC_MODELING = True
RUN_KEYWORD_EXTRACTION = True
RUN_BURNOUT_ANALYSIS = True
SAMPLE_SIZE = 8000


def main():
    start_time = time.time()

    df = load_data("data/mental_health_burnout_tech_2026.csv")

    print("\n====================================")
    print("     MINDSCOPE ANALYTICS PLATFORM")
    print("====================================\n")
    print(f"Original Dataset Size: {len(df):,}")

    if DEVELOPMENT_MODE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"Development Mode: Using {len(df):,} records")
    else:
        print(f"Production Mode: Using {len(df):,} records")

    # ---------------- BUSINESS ANALYTICS ----------------
    avg_burnout = calculate_average_burnout(df)
    top_role, top_score = highest_burnout_role(df)
    avg_depression = sleep_depression_analysis(df)

    print("\n========== BUSINESS INSIGHTS ==========\n")
    for insight in generate_insights(
        avg_burnout, top_role, top_score, avg_depression
    ):
        print("•", insight)

    # ---------------- RISK ANALYSIS ----------------
    workload_burnout = high_workload_risk(df)
    sleep_depression = sleep_risk_analysis(df)
    manager_stress = manager_support_risk(df)

    print("\n========== RISK ANALYSIS ==========\n")
    for risk in generate_risk_insights(
        workload_burnout, sleep_depression, manager_stress
    ):
        print("🚨", risk)

    # ---------------- CORRELATIONS ----------------
    print("\n========== CORRELATION INSIGHTS ==========\n")
    for insight in burnout_correlations(df):
        print("•", insight)

    # ---------------- NLP DEMONSTRATION ----------------
    random.seed(42)
    df["employee_feedback"] = df.apply(generate_feedback, axis=1)
    df["themes"] = df["employee_feedback"].apply(detect_themes)
    df["clean_feedback"] = df["employee_feedback"].apply(preprocess_text)

    nlp_dataset = df[
        [
            "employee_id",
            "burnout_level",
            "burnout_score",
            "employee_feedback",
            "clean_feedback",
            "themes",
        ]
    ].copy()
    nlp_dataset["themes"] = nlp_dataset["themes"].apply(
        lambda values: ", ".join(values)
    )
    nlp_dataset.to_csv("data/nlp_employee_feedback.csv", index=False)

    print("\nNLP dataset saved successfully.")
    print("\n========== NLP THEME INSIGHTS ==========\n")
    for insight in generate_theme_insights(df):
        print("•", insight)

    if RUN_KEYWORD_EXTRACTION:
        print("\n========== TOP NLP KEYWORDS ==========\n")
        for keyword, score in extract_top_keywords(df["clean_feedback"]):
            print(f"{keyword}: {score:.4f}")

    if RUN_TOPIC_MODELING:
        print("\n========== TOPIC MODELING RESULTS ==========\n")
        for topic in extract_topics(df["clean_feedback"]):
            print(f"Topic {topic['topic']}: " + ", ".join(topic["words"]))

    if RUN_BURNOUT_ANALYSIS:
        print("\n========== BURNOUT LANGUAGE ANALYSIS ==========\n")
        for word, count in burnout_word_frequency(df).most_common():
            print(f"{word}: {count}")

    # ---------------- STRUCTURED ML ----------------
    # The predictive model intentionally does NOT use the generated feedback.
    model, preprocessor, accuracy, report, comparison = train_burnout_classifier(df)

    print("\n========== STRUCTURED BURNOUT CLASSIFIER ==========\n")
    print("Model comparison (cross-validated weighted F1):")
    print(comparison.to_string(index=False))
    print(f"\nBest model test accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("\nSaved model: models/burnout_structured_model.pkl")

    elapsed = time.time() - start_time
    print("\n====================================")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print("====================================")


if __name__ == "__main__":
    main()
