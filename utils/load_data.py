"""
src/utils/load_data.py

Goal:
    Load the MTSamples dataset reliably and validate required schema.

This module:
    - Reads data/raw/mtsamples.csv
    - Validates required columns exist
    - Returns a clean DataFrame with standardized column names
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class MTSamplesSchema:
    """
    Defines the required columns for MTSamples-based specialty classification.
    """
    text_col: str = "transcription"
    label_col: str = "medical_specialty"


def load_mtsamples_csv(csv_path: str | Path, schema: MTSamplesSchema = MTSamplesSchema()) -> pd.DataFrame:
    """
    Read mtsamples.csv and validate schema.

    Args:
        csv_path: Path to data/raw/mtsamples.csv
        schema: Required schema definition

    Returns:
        pd.DataFrame with columns: ["text", "label"]

    Raises:
        FileNotFoundError: If csv_path does not exist
        ValueError: If required columns are missing or empty
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in [schema.text_col, schema.label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df = df[[schema.text_col, schema.label_col]].copy()
    df.rename(columns={schema.text_col: "text", schema.label_col: "label"}, inplace=True)

    # Basic cleaning
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().ne("")]
    df = df[df["label"].str.strip().ne("")]

    if df.empty:
        raise ValueError("Dataset is empty after basic filtering (missing/blank text or labels).")

    return df
