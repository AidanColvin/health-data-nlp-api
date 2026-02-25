"""
Goal: Evaluate saved transformer model on test split.
Inputs:
  - models/transformer/distilbert/
  - data/processed/test.csv
Outputs:
  - prints accuracy + macro F1
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import evaluate


def main():
    model_dir = "models/transformer/distilbert"

    test_df = pd.read_csv("data/processed/test.csv")
    ds = Dataset.from_pandas(test_df[["text", "label"]])

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    label2id = model.config.label2id
    if not label2id:
        labels = sorted(test_df["label"].unique().tolist())
        label2id = {l: i for i, l in enumerate(labels)}

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    ds = ds.map(tok, batched=True)
    ds = ds.map(lambda b: {"labels": [label2id[l] for l in b["label"]]}, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    f1 = evaluate.load("f1")
    acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels_np)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels_np, average="macro")["f1"],
        }

    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
    metrics = trainer.evaluate(ds)
    print(metrics)


if __name__ == "__main__":
    main()
