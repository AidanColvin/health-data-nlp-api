"""
Goal: Train RandomForest baseline.
Notes: Not ideal for sparse TF-IDF; included for benchmarking completeness.
"""
from __future__ import annotations
import os, joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.features.vectorize import build_tfidf

def main():
    df = pd.read_csv("data/processed/train.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    pipe = Pipeline([
        ("tfidf", build_tfidf()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X, y)

    os.makedirs("models/classical", exist_ok=True)
    joblib.dump(pipe, "models/classical/rf.joblib")
    print("Saved models/classical/rf.joblib")

if __name__ == "__main__":
    main()
