import re

CUSTOM_STOPWORDS = {
    "recently",
    "lately",
    "past",
    "month",
    "months",
    "experience"
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