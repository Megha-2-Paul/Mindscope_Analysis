def burnout_correlations(df):

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
        "social_support_score",
        "deadline_pressure_score",
        "autonomy_score"
    ]

    corr_matrix = df[numeric_cols].corr()

    burnout_corr = corr_matrix["burnout_score"]

    insights = []

    for feature, corr_value in burnout_corr.items():

        if feature == "burnout_score":
            continue

        feature_name = feature.replace("_", " ").title()

        if corr_value > 0.5:

            insights.append(
                f"{feature_name} shows a strong positive relationship with burnout."
            )

        elif corr_value > 0.3:

            insights.append(
                f"{feature_name} moderately increases burnout trends."
            )

        elif corr_value < -0.5:

            insights.append(
                f"{feature_name} shows a strong negative relationship with burnout."
            )

        elif corr_value < -0.3:

            insights.append(
                f"{feature_name} appears to reduce burnout levels."
            )

    return insights