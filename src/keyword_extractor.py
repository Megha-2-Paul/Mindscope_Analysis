from sklearn.feature_extraction.text import TfidfVectorizer


CUSTOM_STOP_WORDS = [
    "feel",
    "felt",
    "role",
    "work",
    "employee",
    "employees",
    "generally",
    "currently",
    "usually",
    "often"
]


def extract_top_keywords(texts, top_n=15):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=100,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.mean(axis=0).A1

    keyword_scores = list(
        zip(feature_names, scores)
    )

    # Remove custom stop words
    keyword_scores = [
        (word, score)
        for word, score in keyword_scores
        if word not in CUSTOM_STOP_WORDS
    ]

    keyword_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return keyword_scores[:top_n]