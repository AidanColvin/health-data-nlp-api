"""
Goal: Fine-tune a lightweight transformer (DistilBERT) for specialty classification.
Inputs:
  - data/processed/train.csv
  - data/processed/val.csv
Outputs:
  - models/transformer/distilbert/ (HF Trainer output)
Notes:
  - Uses a fixed train/val split (no k-fold CV) for practicality.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


def main():
    model_name = "distilbert-base-uncased"
    out_dir = "models/transformer/distilbert"

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    labels = sorted(train_df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    train_df["label_id"] = train_df["label"].map(label2id)
    val_df["label_id"] = val_df["label"].map(label2id)

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    f1 = evaluate.load("f1")
    acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels_np)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels_np, average="macro")["f1"],
        }

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
