"""
Goal: Provide a shared TF-IDF vectorizer for all classical models.
Inputs: list[str]
Outputs: fitted vectorizer / transformed matrices
Notes: Keep consistent featurization across models for fair comparison.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf():
    return TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        strip_accents="unicode",
        lowercase=True,
    )
