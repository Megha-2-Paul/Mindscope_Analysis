from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def extract_topics(texts, n_topics=5):

    vectorizer = CountVectorizer(
    stop_words="english",
    max_df=0.90,
    min_df=5,
    ngram_range=(1, 2)
)

    doc_term_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()

    topics = []

    for topic_idx, topic in enumerate(lda.components_):

        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-11:-1]
        ]

        topics.append(
            {
                "topic": topic_idx + 1,
                "words": top_words
            }
        )

    return topics