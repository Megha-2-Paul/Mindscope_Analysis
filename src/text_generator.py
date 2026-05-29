import random


HIGH_BURNOUT_OPENINGS = [
    "Over the past few months, the workload has become increasingly difficult to manage.",
    "I have been feeling mentally exhausted due to growing work demands.",
    "The pressure at work has increased significantly and it is starting to affect my wellbeing.",
    "Lately, I have found it difficult to cope with the pace and expectations of my role."
]

MEDIUM_BURNOUT_OPENINGS = [
    "Most days are manageable, although certain periods can become stressful.",
    "I generally enjoy my work, but there are times when workload pressures become noticeable.",
    "My role is rewarding, though balancing responsibilities can sometimes be challenging.",
    "Work is usually manageable, but project deadlines occasionally create additional pressure."
]

LOW_BURNOUT_OPENINGS = [
    "I feel motivated and engaged in my role.",
    "I am generally satisfied with my work environment and responsibilities.",
    "The workload is manageable and allows me to stay productive.",
    "I feel positive about my current role and career growth opportunities."
]


SLEEP_ISSUES = [
    "Poor sleep has affected my concentration and energy levels.",
    "I often feel tired during the workday because I am not getting enough rest.",
    "My sleep schedule has been negatively impacted by work-related stress."
]

STRESS_ISSUES = [
    "Upcoming deadlines often make me feel anxious.",
    "Work-related pressure has become increasingly difficult to manage.",
    "The constant need to meet expectations can be mentally draining."
]

MANAGER_ISSUES = [
    "I would appreciate more support and guidance from management.",
    "Communication from leadership could be improved during demanding periods.",
    "Additional managerial support would help reduce workplace stress."
]

WORK_LIFE_ISSUES = [
    "Maintaining a healthy work-life balance has become challenging.",
    "Work responsibilities often extend into my personal time.",
    "I find it difficult to fully disconnect from work after office hours."
]

POSITIVE_FACTORS = [
    "My manager provides strong support when challenges arise.",
    "The team culture is collaborative and positive.",
    "I feel valued by my team and organization.",
    "I have enough flexibility to manage both work and personal commitments."
]


def generate_feedback(row):

    burnout = row["burnout_score"]
    stress = row["stress_score"]
    sleep = row["sleep_hours_per_night"]
    manager = row["manager_support_score"]
    wlb = row["work_life_balance_score"]
    satisfaction = row["job_satisfaction_score"]

    feedback_parts = []

    # ========================
    # HIGH RISK EMPLOYEE
    # ========================
    if burnout >= 7:

        feedback_parts.append(
            random.choice(HIGH_BURNOUT_OPENINGS)
        )

        if stress >= 7:
            feedback_parts.append(
                random.choice(STRESS_ISSUES)
            )

        if sleep < 6:
            feedback_parts.append(
                random.choice(SLEEP_ISSUES)
            )

        if manager < 4:
            feedback_parts.append(
                random.choice(MANAGER_ISSUES)
            )

        if wlb < 4:
            feedback_parts.append(
                random.choice(WORK_LIFE_ISSUES)
            )

    # ========================
    # MODERATE RISK EMPLOYEE
    # ========================
    elif burnout >= 4:

        feedback_parts.append(
            random.choice(MEDIUM_BURNOUT_OPENINGS)
        )

        if stress >= 6:
            feedback_parts.append(
                random.choice(STRESS_ISSUES)
            )

        if sleep < 6:
            feedback_parts.append(
                random.choice(SLEEP_ISSUES)
            )

        if wlb < 5:
            feedback_parts.append(
                random.choice(WORK_LIFE_ISSUES)
            )

    # ========================
    # HEALTHY EMPLOYEE
    # ========================
    else:

        feedback_parts.append(
            random.choice(LOW_BURNOUT_OPENINGS)
        )

        feedback_parts.append(
            random.choice(POSITIVE_FACTORS)
        )

    # Low satisfaction can affect any employee
    if satisfaction < 4:

        feedback_parts.append(
            "Recently, I have felt less motivated and engaged with my work."
        )

    # Remove duplicates
    feedback_parts = list(dict.fromkeys(feedback_parts))

    return " ".join(feedback_parts)