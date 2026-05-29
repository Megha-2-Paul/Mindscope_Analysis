from collections import Counter
from src.burnout_lexicon import BURNOUT_WORDS


def burnout_word_frequency(df):

    counter = Counter()

    for text in df["employee_feedback"]:

        text = text.lower()

        for word in BURNOUT_WORDS:

            if word in text:
                counter[word] += 1

    return counter