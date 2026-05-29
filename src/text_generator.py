import random

# ==========================================
# INTRODUCTIONS
# ==========================================

INTRODUCTIONS = [
    "Recently,",
    "Over the past few months,",
    "In my current role,",
    "From my experience,",
    "Lately,",
    "Over time,"
]

# ==========================================
# POSITIVE THEMES
# ==========================================

POSITIVE_GROWTH = [
    "I feel that my role provides excellent opportunities for professional growth.",
    "The projects I work on help me develop new skills and stay engaged.",
    "I am satisfied with the learning opportunities available within the organization.",
    "My work allows me to continuously improve my technical and professional skills."
]

POSITIVE_TEAM = [
    "The team culture is collaborative and supportive.",
    "I enjoy working with my colleagues and feel comfortable sharing ideas.",
    "Team members are willing to help each other when challenges arise.",
    "I feel valued as part of a positive and productive team."
]

POSITIVE_FLEXIBILITY = [
    "The flexibility offered by the organization helps me maintain a healthy routine.",
    "I can generally balance work responsibilities with personal commitments.",
    "Remote work options have positively impacted my productivity.",
    "The organization provides enough flexibility to manage work effectively."
]

POSITIVE_MANAGER = [
    "My manager provides strong support whenever challenges arise.",
    "Leadership is approachable and encourages open communication.",
    "I receive useful feedback that helps me improve my performance.",
    "Management creates an environment where employees feel supported."
]

POSITIVE_SATISFACTION = [
    "I feel motivated and engaged in my role.",
    "I am satisfied with my current responsibilities and workload.",
    "My work feels meaningful and rewarding.",
    "I enjoy the challenges that come with my position."
]

# ==========================================
# MEDIUM BURNOUT THEMES
# ==========================================

MEDIUM_WORKLOAD = [
    "There are periods when workload becomes difficult to manage.",
    "Project deadlines can occasionally create additional pressure.",
    "Some weeks require significantly more effort than others.",
    "The pace of work can become challenging during busy periods."
]

MEDIUM_BALANCE = [
    "Balancing work and personal commitments can sometimes be challenging.",
    "Busy periods occasionally affect my ability to disconnect from work.",
    "I sometimes find it difficult to maintain a healthy work-life balance.",
    "Work responsibilities occasionally spill into personal time."
]

MEDIUM_STRESS = [
    "There are times when work-related pressure becomes noticeable.",
    "Meeting deadlines can occasionally be stressful.",
    "Some projects create periods of increased stress and responsibility.",
    "I occasionally feel pressure when managing multiple priorities."
]

MEDIUM_SLEEP = [
    "Stressful periods sometimes affect my sleep quality.",
    "I occasionally struggle to feel fully rested before work.",
    "Busy work schedules can impact my sleep routine.",
    "There are times when work-related thoughts affect my ability to relax."
]

# ==========================================
# HIGH BURNOUT THEMES
# ==========================================

HIGH_EXHAUSTION = [
    "I often feel mentally exhausted by the end of the workday.",
    "The constant pressure has significantly affected my energy levels.",
    "I have noticed increasing signs of burnout over the past few months.",
    "The workload has become overwhelming and difficult to sustain."
]

HIGH_SLEEP = [
    "Poor sleep is affecting my concentration and productivity.",
    "Work-related stress has negatively impacted my sleep quality.",
    "I rarely feel fully rested before beginning work.",
    "Lack of sleep has made it difficult to maintain focus throughout the day."
]

HIGH_STRESS = [
    "The pressure to meet expectations has become mentally draining.",
    "I frequently feel anxious about upcoming deadlines.",
    "Persistent stress is affecting both my motivation and wellbeing.",
    "Managing current responsibilities has become increasingly stressful."
]

HIGH_SUPPORT = [
    "I would benefit from more support from leadership during demanding periods.",
    "Communication from management could be improved.",
    "It is difficult to raise concerns when workload becomes excessive.",
    "Additional managerial support would help reduce workplace stress."
]

HIGH_BALANCE = [
    "Work responsibilities regularly extend into my personal time.",
    "I find it difficult to disconnect from work after office hours.",
    "Maintaining a healthy work-life balance has become increasingly challenging.",
    "The demands of my role leave little time for personal recovery."
]

LOW_SATISFACTION = [
    "Recently, I have felt less motivated and engaged with my work.",
    "My overall job satisfaction has declined over time.",
    "I have become less enthusiastic about my daily responsibilities.",
    "The work feels less rewarding than it previously did."
]

# ==========================================
# MAIN GENERATOR
# ==========================================

def generate_feedback(row):

    burnout = row["burnout_score"]
    stress = row["stress_score"]
    sleep = row["sleep_hours_per_night"]
    manager = row["manager_support_score"]
    wlb = row["work_life_balance_score"]
    satisfaction = row["job_satisfaction_score"]

    feedback_parts = []

    # ======================================
    # HIGH BURNOUT PERSONA
    # ======================================

    if burnout >= 7:

        feedback_parts.append(
            random.choice(HIGH_EXHAUSTION)
        )

        if stress >= 7:
            feedback_parts.append(
                random.choice(HIGH_STRESS)
            )

        if sleep < 6:
            feedback_parts.append(
                random.choice(HIGH_SLEEP)
            )

        if manager < 4:
            feedback_parts.append(
                random.choice(HIGH_SUPPORT)
            )

        if wlb < 4:
            feedback_parts.append(
                random.choice(HIGH_BALANCE)
            )

    # ======================================
    # MEDIUM BURNOUT PERSONA
    # ======================================

    elif burnout >= 4:

        feedback_parts.append(
            random.choice(MEDIUM_WORKLOAD)
        )

        if stress >= 6:
            feedback_parts.append(
                random.choice(MEDIUM_STRESS)
            )

        if sleep < 6:
            feedback_parts.append(
                random.choice(MEDIUM_SLEEP)
            )

        if wlb < 5:
            feedback_parts.append(
                random.choice(MEDIUM_BALANCE)
            )

    # ======================================
    # HEALTHY PERSONA
    # ======================================

    else:

        positive_categories = [
            POSITIVE_GROWTH,
            POSITIVE_TEAM,
            POSITIVE_FLEXIBILITY,
            POSITIVE_MANAGER,
            POSITIVE_SATISFACTION
        ]

        selected_categories = random.sample(
            positive_categories,
            2
        )

        for category in selected_categories:
            feedback_parts.append(
                random.choice(category)
            )

    # ======================================
    # LOW SATISFACTION OVERRIDE
    # ======================================

    if satisfaction < 4:

        feedback_parts.append(
            random.choice(LOW_SATISFACTION)
        )

    # ======================================
    # FINAL ASSEMBLY
    # ======================================

    feedback_parts = list(
        dict.fromkeys(feedback_parts)
    )

    random.shuffle(feedback_parts)

    feedback = (
        random.choice(INTRODUCTIONS)
        + " "
        + " ".join(feedback_parts)
    )

    return feedback