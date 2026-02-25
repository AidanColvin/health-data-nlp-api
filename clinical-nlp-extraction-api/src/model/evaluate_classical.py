"""
evaluate_classical.py

Goal
----
Evaluate the saved classical TF-IDF model on the held-out test split.

Inputs
------
- models/classical/model.joblib
- data/processed/test.csv (columns: text,label)

Outputs
-------
- prints Accuracy and F1 (macro)
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main() -> None:
    test_path = "data/processed/test.csv"
    df = pd.read_csv(test_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("test.csv must have columns: text,label")

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    model = joblib.load("models/classical/model.joblib")
    preds = model.predict(X)

    print("Accuracy:", accuracy_score(y, preds))
    print("F1 (macro):", f1_score(y, preds, average="macro"))


if __name__ == "__main__":
    main()
