"""
src/model/classical/cv_classical.py

Goal:
    Run K-fold cross validation on classical text classifiers (excluding BERT),
    save fold-level + summary metrics, and print a leaderboard to terminal.

Inputs:
    data/processed/train.csv  (must have columns: text, label_id)

Outputs:
    data/processed/metrics_cv_folds.csv
    data/processed/metrics_cv_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class CVConfig:
    k_folds: int = 5
    seed: int = 42
    max_features: int = 50000
    ngram_max: int = 2
    min_df: int = 2
    # SVD is used for tree/boosting models to avoid sparse incompatibilities
    svd_components: int = 300


# ----------------------------
# Data loading
# ----------------------------

def load_train_csv(train_csv: Path) -> Tuple[List[str], np.ndarray]:
    """
    Load training data for CV.

    Args:
        train_csv: path to train.csv with columns: text, label_id

    Returns:
        texts: list[str]
        y: np.ndarray of int label_ids
    """
    train_csv = Path(train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    df = pd.read_csv(train_csv)
    missing = [c for c in ["text", "label_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {train_csv}: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["text", "label_id"]).copy()
    df["text"] = df["text"].astype(str)
    df["label_id"] = df["label_id"].astype(int)

    texts = df["text"].tolist()
    y = df["label_id"].to_numpy(dtype=int)
    if len(texts) < 10:
        raise ValueError(f"Too few rows for CV: {len(texts)}")
    return texts, y


# ----------------------------
# Model builders
# ----------------------------

def _tfidf(cfg: CVConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=cfg.max_features,
        ngram_range=(1, cfg.ngram_max),
        min_df=cfg.min_df,
    )


def build_models(cfg: CVConfig) -> Dict[str, Pipeline]:
    """
    Build classical model pipelines.

    Notes:
        - logreg, svm handle sparse TF-IDF fine.
        - rf, gb often struggle with huge sparse matrices -> use TruncatedSVD to densify.
    """
    models: Dict[str, Pipeline] = {}

    models["logreg"] = Pipeline(
        steps=[
            ("tfidf", _tfidf(cfg)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1)),
        ]
    )

    models["svm"] = Pipeline(
        steps=[
            ("tfidf", _tfidf(cfg)),
            ("clf", LinearSVC()),
        ]
    )

    models["rf"] = Pipeline(
        steps=[
            ("tfidf", _tfidf(cfg)),
            ("svd", TruncatedSVD(n_components=cfg.svd_components, random_state=cfg.seed)),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                random_state=cfg.seed,
                n_jobs=-1,
            )),
        ]
    )

    models["gb"] = Pipeline(
        steps=[
            ("tfidf", _tfidf(cfg)),
            ("svd", TruncatedSVD(n_components=cfg.svd_components, random_state=cfg.seed)),
            ("clf", HistGradientBoostingClassifier(random_state=cfg.seed)),
        ]
    )

    return models


# ----------------------------
# CV runner
# ----------------------------

def run_cv(
    texts: List[str],
    y: np.ndarray,
    models: Dict[str, Pipeline],
    cfg: CVConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run StratifiedKFold CV and return fold-level + summary metrics.

    Metrics:
        - accuracy
        - f1_macro
        - f1_weighted
    """
    skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    fold_rows: List[Dict[str, object]] = []

    for model_name, pipe in models.items():
        fold_idx = 0
        for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
            fold_idx += 1
            X_tr = [texts[i] for i in train_idx]
            X_va = [texts[i] for i in val_idx]
            y_tr = y[train_idx]
            y_va = y[val_idx]

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_va)

            fold_rows.append({
                "model": model_name,
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "accuracy": float(accuracy_score(y_va, y_pred)),
                "f1_macro": float(f1_score(y_va, y_pred, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_va, y_pred, average="weighted", zero_division=0)),
            })

    folds_df = pd.DataFrame(fold_rows)

    summary_df = (
        folds_df
        .groupby("model")[["accuracy", "f1_macro", "f1_weighted"]]
        .agg(["mean", "std"])
    )

    # Flatten columns: ("accuracy","mean") -> "accuracy_mean"
    summary_df.columns = [f"{a}_{b}" for a, b in summary_df.columns.to_list()]
    summary_df = summary_df.reset_index()

    # Leaderboard sort (macro F1 first)
    summary_df = summary_df.sort_values(by=["f1_macro_mean", "accuracy_mean"], ascending=False)

    return folds_df, summary_df


def save_outputs(out_dir: Path, folds_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_path = out_dir / "metrics_cv_folds.csv"
    summary_path = out_dir / "metrics_cv_summary.csv"

    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved: {folds_path}")
    print(f"Saved: {summary_path}\n")


def print_leaderboard(summary_df: pd.DataFrame) -> None:
    cols = [
        "model",
        "accuracy_mean", "accuracy_std",
        "f1_macro_mean", "f1_macro_std",
        "f1_weighted_mean", "f1_weighted_std",
    ]
    view = summary_df[cols].copy()
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    print(view.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/processed/train.csv")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--svd_components", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CVConfig(k_folds=args.k, seed=args.seed, svd_components=args.svd_components)

    texts, y = load_train_csv(Path(args.train_csv))
    models = build_models(cfg)

    folds_df, summary_df = run_cv(texts, y, models, cfg)
    save_outputs(Path(args.out_dir), folds_df, summary_df)
    print("=== CV Leaderboard (sorted by f1_macro_mean) ===")
    print_leaderboard(summary_df)


if __name__ == "__main__":
    main()
