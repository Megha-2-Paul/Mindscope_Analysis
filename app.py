# ==========================================
# IMPORTS
# ==========================================

import time
import random

from src.data_loader import load_data

# Analytics
from src.analysis import (
    calculate_average_burnout,
    highest_burnout_role,
    sleep_depression_analysis,
    high_workload_risk,
    sleep_risk_analysis,
    manager_support_risk
)

# Insight Generators
from src.insight_generator import (
    generate_insights,
    generate_risk_insights
)

# Correlation Engine
from src.correlation_engine import burnout_correlations

# Visualizations
from src.visualization import (
    burnout_by_role,
    workhours_vs_burnout,
    correlation_heatmap,
    theme_frequency_chart
)

# NLP Components
from src.text_generator import generate_feedback
from src.text_preprocessor import preprocess_text
from src.nlp_engine import detect_themes
from src.nlp_insights import generate_theme_insights
from src.keyword_extractor import extract_top_keywords
from src.topic_modeler import extract_topics
from src.nlp_burnout_analysis import burnout_word_frequency

from src.burnout_classifier import (
    train_burnout_classifier
)
# ==========================================
# SYSTEM SETTINGS
# ==========================================

start_time = time.time()

DEVELOPMENT_MODE = True

RUN_VISUALIZATIONS = False
RUN_TOPIC_MODELING = True
RUN_KEYWORD_EXTRACTION = True
RUN_BURNOUT_ANALYSIS = True

SAMPLE_SIZE = 8000


# ==========================================
# LOAD DATA
# ==========================================

df = load_data(
    "data/mental_health_burnout_tech_2026.csv"
)

print("\n====================================")
print("     MINDSCOPE ANALYTICS PLATFORM")
print("====================================\n")

print(f"Original Dataset Size: {len(df):,}")

# ==========================================
# DEVELOPMENT MODE
# ==========================================

if DEVELOPMENT_MODE:

    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

    print(
        f"Development Mode: Using {len(df):,} records"
    )

else:

    print(
        f"Production Mode: Using {len(df):,} records"
    )


# ==========================================
# BUSINESS ANALYTICS
# ==========================================

avg_burnout = calculate_average_burnout(df)

top_role, top_score = highest_burnout_role(df)

avg_depression = sleep_depression_analysis(df)

insights = generate_insights(
    avg_burnout,
    top_role,
    top_score,
    avg_depression
)

print("\n========== BUSINESS INSIGHTS ==========\n")

for insight in insights:
    print("•", insight)


# ==========================================
# RISK ANALYSIS
# ==========================================

workload_burnout = high_workload_risk(df)

sleep_depression = sleep_risk_analysis(df)

manager_stress = manager_support_risk(df)

risk_insights = generate_risk_insights(
    workload_burnout,
    sleep_depression,
    manager_stress
)

print("\n========== RISK ANALYSIS ==========\n")

for risk in risk_insights:
    print("🚨", risk)


# ==========================================
# CORRELATION ANALYSIS
# ==========================================

correlation_insights = burnout_correlations(df)

print("\n========== CORRELATION INSIGHTS ==========\n")

for insight in correlation_insights:
    print("•", insight)


# ==========================================
# NLP FEEDBACK GENERATION
# ==========================================

random.seed(42)

df["employee_feedback"] = df.apply(
    generate_feedback,
    axis=1
)

df["themes"] = df["employee_feedback"].apply(
    detect_themes
)

df["clean_feedback"] = (
    df["employee_feedback"]
    .apply(preprocess_text)
)

nlp_dataset = df[
    [
        "employee_id",
        "burnout_level",
        "burnout_score",
        "employee_feedback",
        "clean_feedback",
        "themes"
    ]
].copy()

nlp_dataset["themes"] = (
    nlp_dataset["themes"]
    .apply(lambda x: ", ".join(x))
)

nlp_dataset.to_csv(
    "data/nlp_employee_feedback.csv",
    index=False
)

print(
    "\nNLP dataset saved successfully."
)

print("\n========== SAMPLE GENERATED FEEDBACK ==========\n")

print(
    df[
        [
            "burnout_score",
            "employee_feedback",
            "themes"
        ]
    ].head(10)
)


# ==========================================
# NLP THEME ANALYSIS
# ==========================================

theme_insights = generate_theme_insights(df)

print("\n========== NLP THEME INSIGHTS ==========\n")

for insight in theme_insights:
    print("•", insight)


# ==========================================
# TF-IDF KEYWORD EXTRACTION
# ==========================================

if RUN_KEYWORD_EXTRACTION:

    top_keywords = extract_top_keywords(
        df["clean_feedback"]
    )

    print("\n========== TOP NLP KEYWORDS ==========\n")

    for keyword, score in top_keywords:

        print(
            f"{keyword}: {score:.4f}"
        )


# ==========================================
# TOPIC MODELING (LDA)
# ==========================================

if RUN_TOPIC_MODELING:

    topics = extract_topics(
        df["clean_feedback"]
    )

    print("\n========== TOPIC MODELING RESULTS ==========\n")

    for topic in topics:

        print(
            f"Topic {topic['topic']}: "
            + ", ".join(topic["words"])
        )


# ==========================================
# BURNOUT LANGUAGE ANALYSIS
# ==========================================

if RUN_BURNOUT_ANALYSIS:

    burnout_words = burnout_word_frequency(df)

    print(
        "\n========== BURNOUT LANGUAGE ANALYSIS ==========\n"
    )

    for word, count in burnout_words.most_common():

        print(
            f"{word}: {count}"
        )


# ==========================================
# VISUALIZATIONS
# ==========================================

if RUN_VISUALIZATIONS:

    # Uncomment if needed
    # burnout_by_role(df)
    # workhours_vs_burnout(df)
    # correlation_heatmap(df)

    theme_frequency_chart(df)

    print(
        "\n========== VISUALIZATIONS ==========\n"
    )

    print(
        "Charts saved to visualizations directory."
    )

# ==========================================
# BURNOUT CLASSIFICATION MODEL
# ==========================================

model, vectorizer, accuracy, report = (
    train_burnout_classifier(df)
)

print(
    "\n========== BURNOUT CLASSIFIER ==========\n"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    "\nClassification Report:\n"
)

print(report)
# ==========================================
# EXECUTION SUMMARY
# ==========================================

end_time = time.time()

print("\n====================================")
print(
    f"Execution Time: "
    f"{end_time - start_time:.2f} seconds"
)
print("====================================")