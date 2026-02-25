"""
src/model/transformer/evaluate_bert_fixed.py

Fix:
  Loads ONLY from a local model directory that exists (no HF repo id confusion).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--model_dir", default="models/transformer/distilbert")
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing: {test_csv}")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Train first:\n"
            f"  python -m src.model.transformer.train_bert_fixed --out_dir {model_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    test_df = pd.read_csv(test_csv).dropna(subset=["text", "label_id"])
    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["label_id"].astype(int).to_numpy()

    y_pred = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=args.max_length, return_tensors="pt")
        out = model(**enc)
        pred = int(torch.argmax(out.logits, dim=1).item())
        y_pred.append(pred)

    y_pred = np.array(y_pred)

    print("accuracy:", accuracy_score(y_true, y_pred))
    print("f1_macro:", f1_score(y_true, y_pred, average="macro", zero_division=0))
    print("f1_weighted:", f1_score(y_true, y_pred, average="weighted", zero_division=0))
    print("\nclassification report:\n", classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
