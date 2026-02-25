"""
evaluate_classical.py

Goal
----
Evaluate the saved classical model on held-out test set.

Inputs
------
- models/classical/model.joblib
- data/processed/test.csv (text,label)

Outputs
-------
- prints Accuracy and Macro F1
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main() -> None:
    df = pd.read_csv("data/processed/test.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    model = joblib.load("models/classical/model.joblib")
    preds = model.predict(X)

    print("Accuracy:", accuracy_score(y, preds))
    print("F1 (macro):", f1_score(y, preds, average="macro"))


if __name__ == "__main__":
    main()
