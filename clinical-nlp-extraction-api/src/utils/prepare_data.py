"""
Goal: Create clean train/val/test splits from raw MTSamples.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.text_cleaning import clean_text
from src.utils.labeling import build_label_maps
import os, json

INPUT = "data/raw/mtsamples.csv"
OUT = "data/processed"

df = pd.read_csv(INPUT)

text_col = "transcription"
label_col = "medical_specialty"

df = df[[text_col, label_col]].dropna()
df[text_col] = df[text_col].apply(clean_text)

train, test = train_test_split(df, test_size=0.1, stratify=df[label_col], random_state=42)
train, val = train_test_split(train, test_size=0.1, stratify=train[label_col], random_state=42)

label2id, id2label = build_label_maps(train[label_col])

os.makedirs(OUT, exist_ok=True)

train.rename(columns={text_col:"text", label_col:"label"}).to_csv(f"{OUT}/train.csv", index=False)
val.rename(columns={text_col:"text", label_col:"label"}).to_csv(f"{OUT}/val.csv", index=False)
test.rename(columns={text_col:"text", label_col:"label"}).to_csv(f"{OUT}/test.csv", index=False)

json.dump({"label2id":label2id,"id2label":id2label}, open(f"{OUT}/labels.json","w"), indent=2)

print("Data prepared.")
