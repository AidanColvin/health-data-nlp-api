"""
Goal: Train Gradient Boosting baseline (HistGradientBoosting).
Notes: Requires dense input; we use TruncatedSVD to reduce TF-IDF to dense.
"""
from __future__ import annotations
import os, joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features.vectorize import build_tfidf

def main():
    df = pd.read_csv("data/processed/train.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    pipe = Pipeline([
        ("tfidf", build_tfidf()),
        ("svd", TruncatedSVD(n_components=300, random_state=42)),
        ("clf", HistGradientBoostingClassifier(random_state=42)),
    ])
    pipe.fit(X, y)

    os.makedirs("models/classical", exist_ok=True)
    joblib.dump(pipe, "models/classical/gb.joblib")
    print("Saved models/classical/gb.joblib")

if __name__ == "__main__":
    main()
