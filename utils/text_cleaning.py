"""
Goal: Normalize clinical text while preserving meaning.
Inputs: raw string
Outputs: cleaned string
"""

import re

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
