def generate_insights(
    avg_burnout,
    top_role,
    top_score,
    avg_depression
):

    insights = []

    insights.append(
        f"Average burnout score across employees is {avg_burnout:.2f}."
    )

    insights.append(
        f"{top_role} reported the highest burnout score at {top_score:.2f}."
    )

    insights.append(
        f"Employees sleeping less than 6 hours show an average PHQ-9 depression score of {avg_depression:.2f}."
    )

    return insights

def generate_risk_insights(
    workload_burnout,
    sleep_depression,
    manager_stress
):

    risks = []

    if workload_burnout > 7:

        risks.append(
            "Employees working over 55 hours weekly are at critical burnout risk."
        )

    if sleep_depression > 10:

        risks.append(
            "Employees sleeping fewer than 5 hours show severe depression indicators."
        )

    if manager_stress > 7:

        risks.append(
            "Low manager support is strongly associated with elevated stress levels."
        )

    return risks