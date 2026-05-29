from collections import Counter


def generate_theme_insights(df):

    counter = Counter()

    for themes in df["themes"]:

        counter.update(themes)

    total_employees = len(df)

    insights = []

    for theme, count in counter.most_common():

        percentage = (count / total_employees) * 100

        insights.append(
            f"{theme} was mentioned by {percentage:.1f}% of employees."
        )

    return insights