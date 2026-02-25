"""
src/model/evaluate.py

Goal:
    Evaluate the trained model on test.csv and print metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model_dir", required=True, help="models/specialty_classifier")
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    test_df = pd.read_csv(data_dir / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["label_id"].astype(int).to_numpy()

    y_pred = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        out = model(**enc)
        pred = int(torch.argmax(out.logits, dim=1).item())
        y_pred.append(pred)

    y_pred = np.array(y_pred)

    print("accuracy:", accuracy_score(y_true, y_pred))
    print("f1_macro:", f1_score(y_true, y_pred, average="macro"))
    print("f1_weighted:", f1_score(y_true, y_pred, average="weighted"))
    print("\nclassification report:\n", classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
