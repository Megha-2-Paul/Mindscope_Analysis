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