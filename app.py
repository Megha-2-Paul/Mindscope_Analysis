from src.data_loader import load_data

from src.analysis import (
    calculate_average_burnout,
    highest_burnout_role,
    sleep_depression_analysis
)

from src.correlation_engine import burnout_correlations

from src.insight_generator import generate_insights

from src.visualization import (
    burnout_by_role,
    workhours_vs_burnout,
    correlation_heatmap
)

from src.analysis import (
    high_workload_risk,
    sleep_risk_analysis,
    manager_support_risk
)

from src.insight_generator import generate_risk_insights

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

# Risk analysis
workload_burnout = high_workload_risk(df)

sleep_depression = sleep_risk_analysis(df)

manager_stress = manager_support_risk(df)

# Generate risk insights
risk_insights = generate_risk_insights(
    workload_burnout,
    sleep_depression,
    manager_stress
)

# Print risk insights
for risk in risk_insights:
    print("🚨", risk)

# Correlation intelligence
correlation_insights = burnout_correlations(df)

print("\n--- Correlation Insights ---\n")

for insight in correlation_insights:
    print("•", insight)
    
# Visualization
burnout_by_role(df)
workhours_vs_burnout(df)
correlation_heatmap(df)