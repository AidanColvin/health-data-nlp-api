"""
src/model/dataset.py

Goal:
    Provide a PyTorch Dataset for transformer fine-tuning.

Outputs:
    - input_ids
    - attention_mask
    - labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TokenizationConfig:
    max_length: int = 256


class ClinicalNotesDataset(Dataset):
    """
    Simple dataset for text classification.

    Args:
        texts: List of note strings
        labels: List of integer label ids
        tokenizer: HF tokenizer
        cfg: tokenization config
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        cfg: TokenizationConfig = TokenizationConfig(),
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item
