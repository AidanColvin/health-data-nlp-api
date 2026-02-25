"""
Goal: Train Linear SVM classifier using TF-IDF.
Inputs: data/processed/train.csv
Outputs: models/classical/svm.joblib
Notes: Often best classical model for sparse TF-IDF text.
"""
from __future__ import annotations
import os, joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.features.vectorize import build_tfidf

def main():
    df = pd.read_csv("data/processed/train.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    pipe = Pipeline([
        ("tfidf", build_tfidf()),
        ("clf", LinearSVC(class_weight="balanced")),
    ])
    pipe.fit(X, y)

    os.makedirs("models/classical", exist_ok=True)
    joblib.dump(pipe, "models/classical/svm.joblib")
    print("Saved models/classical/svm.joblib")

if __name__ == "__main__":
    main()
