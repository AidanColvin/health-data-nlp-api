"""
src/utils/prepare_data.py

CLI wrapper around preprocessing steps.

Example:
    python src/utils/prepare_data.py --input data/raw/mtsamples.csv --out_dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.load_data import load_mtsamples_csv
from src.utils.preprocess import preprocess_and_save, SplitConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to mtsamples.csv")
    p.add_argument("--out_dir", required=True, help="Output directory for processed splits")
    p.add_argument("--train_size", type=float, default=0.80)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--test_size", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_mtsamples_csv(args.input)

    cfg = SplitConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )
    preprocess_and_save(df, out_dir=Path(args.out_dir), cfg=cfg)
    print(f"Saved processed data to: {args.out_dir}")


if __name__ == "__main__":
    main()
