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


def split_dataframe(df, cfg):
    """
    Split dataframe into train/val/test.

    Robust behavior:
      - Attempts stratified splits when feasible.
      - If stratification fails (rare classes or unlucky split), falls back to non-stratified split.
      - Supports cfg.val_size == 0.0 by returning an empty val split.

    Args:
        df: DataFrame with columns ["text", "label", ...]
        cfg: SplitConfig with train_size, val_size, test_size, random_state

    Returns:
        (train_df, val_df, test_df) with reset indexes.
    """
    from sklearn.model_selection import train_test_split

    train_size = float(getattr(cfg, "train_size", 0.8))
    val_size = float(getattr(cfg, "val_size", 0.1))
    test_size = float(getattr(cfg, "test_size", 0.1))
    rs = int(getattr(cfg, "random_state", 42))

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split sizes must sum to 1.0. Got train+val+test={total:.6f}")

    df = df.reset_index(drop=True)

    temp_size = val_size + test_size
    if temp_size <= 0.0:
        empty = df.iloc[0:0].copy()
        return df.copy(), empty, empty

    strat1 = None
    if "label" in df.columns:
        vc = df["label"].value_counts()
        if len(vc) > 0 and vc.min() >= 2:
            strat1 = df["label"]

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_size,
            random_state=rs,
            stratify=strat1,
        )
    except ValueError:
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_size,
            random_state=rs,
            stratify=None,
        )

    if val_size <= 0.0:
        val_df = df.iloc[0:0].copy()
        test_df = temp_df
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    val_ratio_of_temp = val_size / temp_size
    second_test_size = 1.0 - val_ratio_of_temp

    if second_test_size <= 0.0:
        val_df = temp_df
        test_df = temp_df.iloc[0:0].copy()
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    if second_test_size >= 1.0:
        val_df = temp_df.iloc[0:0].copy()
        test_df = temp_df
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    strat2 = None
    if "label" in temp_df.columns:
        vc2 = temp_df["label"].value_counts()
        if len(vc2) > 0 and vc2.min() >= 2:
            strat2 = temp_df["label"]

    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=second_test_size,
            random_state=rs,
            stratify=strat2,
        )
    except ValueError:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=second_test_size,
            random_state=rs,
            stratify=None,
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

