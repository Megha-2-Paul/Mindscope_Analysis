THEME_KEYWORDS = {
    "Workload": [
        "workload",
        "deadline",
        "deadlines",
        "projects",
        "expectations",
        "pressure",
        "pace"
    ],

    "Sleep": [
        "sleep",
        "rest",
        "tired",
        "energy",
        "concentration"
    ],

    "Stress": [
        "stress",
        "stressed",
        "anxious",
        "pressure",
        "drained",
        "exhausted"
    ],

    "Management": [
        "manager",
        "management",
        "leadership",
        "support",
        "guidance",
        "feedback"
    ],

    "Work-Life Balance": [
        "work-life",
        "personal time",
        "disconnect",
        "balance",
        "commitments"
    ]
}


def detect_themes(text):

    text = text.lower()

    detected_themes = []

    for theme, keywords in THEME_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:

                detected_themes.append(theme)
                break

    return detected_themes