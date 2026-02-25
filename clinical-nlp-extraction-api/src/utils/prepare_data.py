"""
prepare_data.py

Goal
----
Prepare MTSamples into reproducible train/val/test splits for specialty classification.

Inputs
------
- --input: path to mtsamples.csv
- --out_dir: output directory for processed splits
- Optional: --text_col, --label_col if your CSV schema differs

Outputs
-------
- <out_dir>/train.csv
- <out_dir>/val.csv
- <out_dir>/test.csv
- <out_dir>/labels.json

Notes
-----
- Stratified splits.
- Removes rows with missing text/labels.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.text_cleaning import clean_text
from src.utils.labeling import build_label_maps


def _guess_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    text_candidates = ["transcription", "text", "note", "report"]
    label_candidates = ["medical_specialty", "specialty", "label", "category"]

    text_col = next((cols[c] for c in text_candidates if c in cols), None)
    label_col = next((cols[c] for c in label_candidates if c in cols), None)

    if not text_col or not label_col:
        raise ValueError(
            "Could not auto-detect text/label columns. "
            f"Columns found: {list(df.columns)}. "
            "Provide --text_col and --label_col."
        )
    return text_col, label_col


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/mtsamples.csv")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--text_col", default=None)
    p.add_argument("--label_col", default=None)
    p.add_argument("--test_size", type=float, default=0.10)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.input)

    guessed_text, guessed_label = _guess_columns(df)
    text_col = args.text_col or guessed_text
    label_col = args.label_col or guessed_label

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).map(clean_text)
    df[label_col] = df[label_col].astype(str).str.strip()

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df[label_col], random_state=args.seed
    )
    val_rel = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_rel, stratify=train_df[label_col], random_state=args.seed
    )

    label2id, id2label = build_label_maps(train_df[label_col].tolist())

    os.makedirs(args.out_dir, exist_ok=True)
    train_df.rename(columns={text_col: "text", label_col: "label"}).to_csv(
        os.path.join(args.out_dir, "train.csv"), index=False
    )
    val_df.rename(columns={text_col: "text", label_col: "label"}).to_csv(
        os.path.join(args.out_dir, "val.csv"), index=False
    )
    test_df.rename(columns={text_col: "text", label_col: "label"}).to_csv(
        os.path.join(args.out_dir, "test.csv"), index=False
    )

    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print("Wrote:", os.path.join(args.out_dir, "train.csv"))
    print("Wrote:", os.path.join(args.out_dir, "val.csv"))
    print("Wrote:", os.path.join(args.out_dir, "test.csv"))
    print("Wrote:", os.path.join(args.out_dir, "labels.json"))


if __name__ == "__main__":
    main()
