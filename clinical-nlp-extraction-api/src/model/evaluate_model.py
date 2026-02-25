"""
Goal: Evaluate any saved sklearn pipeline on a given split.
Inputs:
  --model_path: joblib file for sklearn Pipeline
  --data_csv: CSV with columns text,label
Outputs:
  JSON metrics printed and optionally saved to --out_json
"""
from __future__ import annotations
import argparse, json, os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_csv", required=True)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.data_csv)
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    model = joblib.load(args.model_path)
    preds = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "f1_weighted": float(f1_score(y, preds, average="weighted")),
        "precision_macro": float(precision_score(y, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, preds, average="macro", zero_division=0)),
        "n": int(len(y)),
    }

    print(json.dumps(metrics, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
