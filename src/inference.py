from typing import Dict, Tuple

def predict(text: str) -> Tuple[str, Dict, float]:
    return "Unknown", {"symptoms": [], "negated_symptoms": []}, 0.0
