"""
src/api/main.py

Goal:
    FastAPI inference endpoint for clinical specialty classification.

Assumptions:
    - A trained model exists at models/specialty_classifier
    - label_map.json saved alongside model
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_ROOT = Path(__file__).resolve().parents[2]  # repo_root/src/api/main.py -> repo_root
DEFAULT_MODEL_DIR = APP_ROOT / "models" / "specialty_classifier"

app = FastAPI(title="Clinical NLP Extraction API", version="1.0.0")


class PredictRequest(BaseModel):
    transcription: str


class PredictResponse(BaseModel):
    specialty: str
    confidence_score: float


def _load_label_map(model_dir: Path) -> Dict[str, int]:
    p = model_dir / "label_map.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing label_map.json at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in label_map.items()}


# Lazy globals
_tokenizer = None
_model = None
_id_to_label = None


def _ensure_loaded(model_dir: Path = DEFAULT_MODEL_DIR) -> None:
    global _tokenizer, _model, _id_to_label
    if _tokenizer is not None and _model is not None and _id_to_label is not None:
        return

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    label_map = _load_label_map(model_dir)
    _id_to_label = _invert_label_map(label_map)

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _model.eval()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest) -> PredictResponse:
    try:
        _ensure_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    text = req.transcription.strip()
    if not text:
        raise HTTPException(status_code=400, detail="transcription must be non-empty")

    enc = _tokenizer(text, truncation=True, max_length=256, padding=True, return_tensors="pt")
    out = _model(**enc)
    probs = torch.softmax(out.logits, dim=1).squeeze(0)
    pred_id = int(torch.argmax(probs).item())
    conf = float(probs[pred_id].item())

    specialty = _id_to_label.get(pred_id, "UNKNOWN")
    return PredictResponse(specialty=specialty, confidence_score=conf)
