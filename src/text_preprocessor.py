import re

CUSTOM_STOPWORDS = {
    "work",
    "feel",
    "felt",
    "role",
    "employee",
    "employees",
    "generally",
    "currently",
    "usually",
    "recently",
    "past",
    "months",
    "month",
    "recently",
    "lately",
    "time",
    "current",
    "experience",
    "role",
    "work",
    "employee",
    "employees",
    "generally",
    "currently",
    "usually",
    "felt",
    "feel"
}


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in CUSTOM_STOPWORDS
    ]

    return " ".join(words)