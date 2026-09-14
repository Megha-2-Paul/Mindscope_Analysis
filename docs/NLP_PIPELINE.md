# NLP Pipeline

## Purpose

The NLP component explores employee feedback for recurring themes, keywords, topics, and burnout-related language.

## Current pipeline

```text
Feedback text
    |
    v
Text preprocessing
    |
    +--> Theme detection
    +--> TF-IDF keyword extraction
    +--> LDA topic modelling
    +--> Burnout-language frequency analysis
```

## Synthetic feedback

`src/text_generator.py` can create demonstration feedback from structured employee attributes. This is useful for testing the NLP workflow when real text is unavailable.

Because the generator uses burnout-related attributes, generated feedback must not be treated as independent evidence or used as the primary training source for burnout prediction.

## Theme detection

`src/nlp_engine.py` identifies predefined or lexicon-based themes from feedback. `src/nlp_insights.py` converts detected themes into higher-level observations.

## Keyword extraction

`src/keyword_extractor.py` uses TF-IDF to identify terms that are relatively informative within the feedback corpus.

## Topic modelling

`src/topic_modeler.py` uses Latent Dirichlet Allocation (LDA) to discover groups of words that frequently occur together. Topics are exploratory and require human interpretation.

## Burnout-language analysis

`src/nlp_burnout_analysis.py` counts burnout-related language using the project's lexicon and processed feedback.

## Future improvements

- Support real, consented feedback datasets
- Add sentiment analysis with validation
- Improve theme taxonomy
- Compare rule-based themes with embedding-based clustering
- Add topic-quality metrics and stability checks
- Add privacy-preserving text handling
- Link NLP signals to structured analysis without creating target leakage
- Add explainable examples for why a theme was detected
