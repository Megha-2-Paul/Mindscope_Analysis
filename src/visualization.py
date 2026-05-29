import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter



def burnout_by_role(df):

    role_burnout = (
        df.groupby("job_role")["burnout_score"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))

    role_burnout.plot(kind="bar")

    plt.title("Average Burnout Score by Job Role")

    plt.xlabel("Job Role")

    plt.ylabel("Burnout Score")

    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save chart
    plt.savefig(
        "visualizations/burnout_by_role.png"
    )
    
    plt.close()
    
#Work Hours vs Burnout Scatter Plot
def workhours_vs_burnout(df):

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["work_hours_per_week"],
        df["burnout_score"],
        alpha=0.5
    )

    plt.title("Work Hours vs Burnout")

    plt.xlabel("Work Hours Per Week")

    plt.ylabel("Burnout Score")

    plt.tight_layout()

    plt.savefig(
        "visualizations/workhours_vs_burnout.png"
    )

    plt.close()
    
#Correlation Heatmap
def correlation_heatmap(df):

    numeric_cols = [
        "stress_score",
        "burnout_score",
        "phq9_score",
        "gad7_score",
        "work_hours_per_week",
        "sleep_hours_per_night",
        "manager_support_score",
        "work_life_balance_score",
        "job_satisfaction_score",
        "social_support_score"
    ]

    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Mental Health Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "visualizations/correlation_heatmap.png"
    )

    plt.close()
    
print("\nCharts saved to Visualization directory")


def theme_frequency_chart(df):

    counter = Counter()

    for themes in df["themes"]:
        counter.update(themes)

    plt.figure(figsize=(8, 5))

    plt.bar(
        counter.keys(),
        counter.values()
    )

    plt.title("Employee Feedback Themes")

    plt.xlabel("Theme")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "visualizations/theme_frequency.png"
    )

    plt.close()