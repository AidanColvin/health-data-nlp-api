"""
train_classical.py

Goal
----
Train and compare strong classical baselines for specialty classification using TF-IDF features.

Inputs
------
- data/processed/train.csv (columns: text,label)

Outputs
-------
- models/classical/model.joblib
- models/classical/metrics.json

Notes
-----
- Uses StratifiedKFold cross-validation (5-fold).
- Trains Logistic Regression and Linear SVM (best-practice baselines for text).
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def main() -> None:
    train_path = "data/processed/train.csv"
    df = pd.read_csv(train_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("train.csv must have columns: text,label")

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
    }

    results = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=2)),
                ("clf", clf),
            ]
        )

        preds = cross_val_predict(pipe, X, y, cv=cv)
        score = f1_score(y, preds, average="macro")
        results[name] = {"f1_macro_cv": float(score)}

        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipe

    assert best_pipeline is not None and best_name is not None

    best_pipeline.fit(X, y)

    os.makedirs("models/classical", exist_ok=True)
    joblib.dump(best_pipeline, "models/classical/model.joblib")

    with open("models/classical/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"cv_results": results, "best_model": best_name}, f, indent=2)

    print("Training complete.")
    print("CV results:", results)
    print("Best model:", best_name, "f1_macro_cv=", best_score)
    print("Saved: models/classical/model.joblib")
    print("Saved: models/classical/metrics.json")


if __name__ == "__main__":
    main()
