"""
src/model/transformer/train_bert_fixed.py

Fixes:
  1) transformers v5.x uses TrainingArguments(eval_strategy=...) not evaluation_strategy.
  2) Ensures the output directory exists and is saved as a local path.

Requires:
  data/processed/train.csv and data/processed/val.csv with columns: text, label_id
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--model_name",     p.add_argument("--model_name",     p.add_argument("--model_name",     p.adde    p.add_argument("--model_n   p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    label_map_path = data_dir / "label_map.json"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing: {train_csv}. Run prepare_data first.")
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing: {val_csv}. Run prepare_data with val_size > 0.")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing: {label_map_path}. Run prepare_data first.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # GOTCHA #2 fixed

    train_df = pd.read_csv(train_csv).dropna(subset=["text", "label_id"])
    val_df = pd.read_csv(val_csv).dropna(subset=["text", "label_id"])

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    num_labels = len(label_map)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    train_ds = TextDataset(
        train_df["text"].astype(str).tolist(),
        train_df["label_id"].astype(int).to_numpy(),
        tokenizer,
        args.max_length,
    )
    val_ds = TextDataset(
        val_df["text"].astype(str).tolist(),
        val_df["label_id"].astype(int).to_numpy(),
        tokenizer,
        args.max_length,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "runs"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",  # GOTCHA #1 fixed (NOT evaluation_strategy)
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],  # avoids needing wandb etc.
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save to local model directory (GOTCHA #2)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved local model to: {out_dir}\n")


if __name__ == "__main__":
    main()
