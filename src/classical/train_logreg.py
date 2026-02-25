"""
Goal: Train Logistic Regression text classifier using TF-IDF.
Inputs: data/processed/train.csv
Outputs: models/classical/logreg.joblib
Notes: Strong baseline for text; fast and reliable.
"""
from __future__ import annotations
import os, joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.features.vectorize import build_tfidf

def main():
    df = pd.read_csv("data/processed/train.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    pipe = Pipeline([
        ("tfidf", build_tfidf()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X, y)

    os.makedirs("models/classical", exist_ok=True)
    joblib.dump(pipe, "models/classical/logreg.joblib")
    print("Saved models/classical/logreg.joblib")

if __name__ == "__main__":
    main()
