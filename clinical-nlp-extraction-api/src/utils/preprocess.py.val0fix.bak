"""
src/utils/preprocess.py

Goal:
    Produce an ML-ready dataset for specialty classification with:
    - Cleaned text
    - Label encoding
    - Train/val/test split saved to disk

Notes:
    - No leakage: splits happen before any model training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for dataset splitting.

    train_size + val_size + test_size must equal 1.0
    """
    train_size: float = 0.80
    val_size: float = 0.10
    test_size: float = 0.10
    random_state: int = 42
    min_class_count: int = 2


def clean_clinical_text(text: str) -> str:
    """
    Lightweight, safe text normalization for clinical notes.

    Args:
        text: raw transcription

    Returns:
        cleaned text
    """
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    # Keep punctuation (BERT-style models generally handle it fine); just normalize spacing
    return t


def build_label_map(labels: pd.Series) -> Dict[str, int]:
    """
    Build a stable label->id mapping.

    Args:
        labels: Series of label strings

    Returns:
        dict mapping label string to integer id
    """
    uniq = sorted(labels.unique().tolist())
    return {lab: i for i, lab in enumerate(uniq)}


def split_dataframe(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test with stratification.

    Args:
        df: DataFrame with columns ["text","label"]
        cfg: split config

    Returns:
        train_df, val_df, test_df
    """
    if round(cfg.train_size + cfg.val_size + cfg.test_size, 6) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - cfg.train_size),
        random_state=cfg.random_state,
        stratify=df["label"],
    )

    # val/test split from temp
    val_ratio_of_temp = cfg.val_size / (cfg.val_size + cfg.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_of_temp),
        random_state=cfg.random_state,        stratify=(temp_df["label"] if temp_df["label"].value_counts().min() >= 2 else None),
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def preprocess_and_save(
    df: pd.DataFrame,
    out_dir: str | Path,
    cfg: SplitConfig = SplitConfig(),
) -> None:
    """
    Create ML-ready splits + label map, and save to disk as CSV + JSON.

    Writes:
        out_dir/train.csv
        out_dir/val.csv
        out_dir/test.csv
        out_dir/label_map.json

    Args:
        df: raw loaded DataFrame with ["text","label"]
        out_dir: directory to write processed files
        cfg: split configuration
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["text"] = df["text"].map(clean_clinical_text)

    # Drop labels that are too rare to stratify reliably
    label_counts = df["label"].value_counts()
    keep = label_counts[label_counts >= cfg.min_class_count].index
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No rows remaining after filtering rare labels. "
            "Lower min_class_count or inspect your label distribution."
        )

    train_df, val_df, test_df = split_dataframe(df, cfg)

    label_map = build_label_map(train_df["label"])  # map derived from train only (good practice)
    for split_df in (train_df, val_df, test_df):
        split_df["label_id"] = split_df["label"].map(label_map)
        if split_df["label_id"].isna().any():
            # If a label appears only in val/test, drop it (rare but possible).
            split_df.dropna(subset=["label_id"], inplace=True)
            split_df["label_id"] = split_df["label_id"].astype(int)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

