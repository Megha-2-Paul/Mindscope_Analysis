from src.data_loader import load_data

from src.analysis import (
    calculate_average_burnout,
    highest_burnout_role,
    sleep_depression_analysis
)

from src.insight_generator import generate_insights


# Load dataset
df = load_data(
    "data/mental_health_burnout_tech_2026.csv"
)

# Run analysis
avg_burnout = calculate_average_burnout(df)

top_role, top_score = highest_burnout_role(df)

avg_depression = sleep_depression_analysis(df)

# Generate insights
insights = generate_insights(
    avg_burnout,
    top_role,
    top_score,
    avg_depression
)

# Print insights
for insight in insights:
    print("•", insight)