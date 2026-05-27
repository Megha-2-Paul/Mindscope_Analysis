def calculate_average_burnout(df):
    return df["burnout_score"].mean()


def highest_burnout_role(df):

    role_burnout = (
        df.groupby("job_role")["burnout_score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_role = role_burnout.index[0]
    top_score = role_burnout.iloc[0]

    return top_role, top_score


def sleep_depression_analysis(df):

    low_sleep = df[df["sleep_hours_per_night"] < 6]

    avg_depression = low_sleep["phq9_score"].mean()

    return avg_depression