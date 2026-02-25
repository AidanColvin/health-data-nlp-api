"""
Goal: Evaluate all model artifacts on test split and write a leaderboard.
Inputs:
  - data/processed/test.csv
  - models in models/classical/*
Outputs:
  - reports/leaderboard.json
  - prints sorted results
"""
from __future__ import annotations
import json, os, subprocess, sys

MODELS = {
    "logreg": "models/classical/logreg.joblib",
    "svm": "models/classical/svm.joblib",
    "rf": "models/classical/rf.joblib",
    "gb": "models/classical/gb.joblib",
}

def eval_one(name: str, path: str) -> dict:
    cmd = [
        sys.executable, "-m", "src.model.evaluate_model",
        "--model_path", path,
        "--data_csv", "data/processed/test.csv",
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def main():
    results = []
    for name, path in MODELS.items():
        if not os.path.exists(path):
            continue
        m = eval_one(name, path)
        m["model"] = name
        m["artifact"] = path
        results.append(m)

    results.sort(key=lambda x: x["f1_macro"], reverse=True)

    os.makedirs("reports", exist_ok=True)
    with open("reports/leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for r in results:
        print(r["model"], "f1_macro=", r["f1_macro"], "acc=", r["accuracy"])

if __name__ == "__main__":
    main()
