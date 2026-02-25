"""
train_classical.py

Goal
----
Train and compare TF-IDF baselines with 5-fold Stratified CV, then fit best model on full train split.

Inputs
------
- data/processed/train.csv (text,label)

Outputs
-------
- models/classical/model.joblib
- models/classical/metrics.json

Notes
-----
- Strong baselines for text: Logistic Regression + Linear SVM.
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
    df = pd.read_csv("data/processed/train.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    best_name = None
    best_score = -1.0

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

    assert best_name is not None

    # Fit best on full training split
    best_clf = models[best_name]
    best_pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=2)),
            ("clf", best_clf),
        ]
    )
    best_pipe.fit(X, y)

    os.makedirs("models/classical", exist_ok=True)
    joblib.dump(best_pipe, "models/classical/model.joblib")

    with open("models/classical/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"cv_results": results, "best_model": best_name}, f, indent=2)

    print("Training complete.")
    print("CV results:", results)
    print("Best model:", best_name, "f1_macro_cv=", best_score)
    print("Saved: models/classical/model.joblib")
    print("Saved: models/classical/metrics.json")


if __name__ == "__main__":
    main()
