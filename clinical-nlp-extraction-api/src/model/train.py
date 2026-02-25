"""
src/model/train.py

Goal:
    Fine-tune a transformer model for MTSamples specialty classification.

This script:
    - Loads processed splits from data/processed
    - Fine-tunes AutoModelForSequenceClassification
    - Saves model + tokenizer into models/

Example:
    python src/model/train.py --data_dir data/processed --model_dir models --model_name emilyalsentzer/Bio_ClinicalBERT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

from src.model.dataset import ClinicalNotesDataset, TokenizationConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Directory containing train.csv/val.csv/label_map.json")
    p.add_argument("--model_dir", required=True, help="Directory to save trained model artifacts")
    p.add_argument("--model_name", default="emilyalsentzer/Bio_ClinicalBERT", help="HF model name")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    label_map_path = data_dir / "label_map.json"

    for p in [train_path, val_path, label_map_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    num_labels = len(label_map)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    tok_cfg = TokenizationConfig(max_length=args.max_length)

    train_ds = ClinicalNotesDataset(
        texts=train_df["text"].astype(str).tolist(),
        labels=train_df["label_id"].astype(int).tolist(),
        tokenizer=tokenizer,
        cfg=tok_cfg,
    )
    val_ds = ClinicalNotesDataset(
        texts=val_df["text"].astype(str).tolist(),
        labels=val_df["label_id"].astype(int).tolist(),
        tokenizer=tokenizer,
        cfg=tok_cfg,
    )

    out_path = model_dir / "specialty_classifier"
    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save final artifacts
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # Save label map next to model
    with open(out_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
