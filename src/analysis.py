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

#Burnout Risk Detection
def high_workload_risk(df):

    high_workload = df[df["work_hours_per_week"] > 55]

    avg_burnout = high_workload["burnout_score"].mean()

    return avg_burnout

#Sleep Risk Detection
def sleep_risk_analysis(df):

    poor_sleep = df[df["sleep_hours_per_night"] < 5]

    avg_phq9 = poor_sleep["phq9_score"].mean()

    return avg_phq9

#Manager Support Risk
def manager_support_risk(df):

    low_support = df[df["manager_support_score"] < 4]

    avg_stress = low_support["stress_score"].mean()

    return avg_stress