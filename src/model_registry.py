"""
Goal: Define canonical paths for trained artifacts.
Notes: Lets compare + API load best model consistently.
"""
from __future__ import annotations

CLASSICAL_DIR = "models/classical"
TRANSFORMER_DIR = "models/transformer"

CLASSICAL_ARTIFACTS = {
    "logreg": f"{CLASSICAL_DIR}/logreg.joblib",
    "svm": f"{CLASSICAL_DIR}/svm.joblib",
    "rf": f"{CLASSICAL_DIR}/rf.joblib",
    "gb": f"{CLASSICAL_DIR}/gb.joblib",
}

TRANSFORMER_ARTIFACTS = {
    "distilbert": f"{TRANSFORMER_DIR}/distilbert",
}
