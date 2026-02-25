from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

ROOT = Path.cwd()
DATA = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
OUT_PDF = ROOT / "MODEL_REPORT.pdf"

def must_exist(p: Path, label: str) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Missing {label}: {p}")
    return p

def safe_read_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    return pd.read_csv(p)

def save_bar(df: pd.DataFrame, x: str, y: str, title: str, out: Path):
    if df is None or df.empty:
        return
    plt.figure()
    dd = df.copy()
    dd = dd.sort_values(y, ascending=False)
    plt.bar(dd[x].astype(str), dd[y].astype(float))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()

def save_heatmap_cm(cm: np.ndarray, labels: list[str], title: str, out: Path, max_labels: int = 25):
    # If too many classes, show top-K by support order already in label_map (or first K)
    k = min(len(labels), max_labels)
    cm2 = cm[:k, :k]
    labs = labels[:k]

    plt.figure(figsize=(8, 7))
    plt.imshow(cm2, aspect="auto")
    plt.title(title + f" (top {k} labels)")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks(range(k), labs, rotation=90, fontsize=7)
    plt.yticks(range(k), labs, fontsize=7)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()

def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, w: float, leading: float = 13) -> float:
    words = text.split()
    line = []
    for word in words:
        trial = (" ".join(line + [word])).strip()
        if c.stringWidth(trial, "Helvetica", 10) <= w:
            line.append(word)
        else:
            c.drawString(x, y, " ".join(line))
            y -= leading
            line = [word]
    if line:
        c.drawString(x, y, " ".join(line))
        y -= leading
    return y

def add_table_text(c: canvas.Canvas, df: pd.DataFrame, x: float, y: float, w: float, max_rows: int = 15) -> float:
    c.setFont("Helvetica", 9)
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    # simple fixed-width-ish text table
    header = " | ".join([str(cn) for cn in cols])
    y = draw_wrapped(c, header, x, y, w, leading=11)
    y -= 2
    c.setLineWidth(0.5)
    c.line(x, y, x + w, y)
    y -= 10
    for _, r in show.iterrows():
        row = " | ".join([f"{r[c]:.6g}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols])
        y = draw_wrapped(c, row, x, y, w, leading=11)
        if y < 1.0 * inch:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = 10.5 * inch
    return y

def main():
    # Inputs from your process
    cv_summary = safe_read_csv(DATA / "metrics_cv_summary.csv")
    cv_folds = safe_read_csv(DATA / "metrics_cv_folds.csv")
    leaderboard = safe_read_csv(DATA / "leaderboard.csv")  # produced by compare_models
    label_map_path = DATA / "label_map.json"
    label_map = None
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    # If you already created a combined leaderboard csv, use it; else build from what exists.
    combined_path = DATA / "combined_leaderboard.csv"
    combined = safe_read_csv(combined_path)
    if combined is None:
        parts = []
        if cv_summary is not None:
            tmp = cv_summary.copy()
            tmp["split"] = "cv5_mean"
            tmp["source"] = "classical_cv"
            parts.append(tmp[["model","split","accuracy_mean","f1_macro_mean","source"]].rename(
                columns={"accuracy_mean":"accuracy","f1_macro_mean":"f1_macro"}
            ))
        if leaderboard is not None:
            # leaderboard.csv might have different cols; normalize if possible
            tmp = leaderboard.copy()
            if "model" in tmp.columns and ("accuracy" in tmp.columns or "acc" in tmp.columns) and ("f1_macro" in tmp.columns):
                if "acc" in tmp.columns and "accuracy" not in tmp.columns:
                    tmp["accuracy"] = tmp["acc"]
                tmp["split"] = "test"
                tmp["source"] = "classical_test"
                parts.append(tmp[["model","split","accuracy","f1_macro","source"]])
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["model","split","accuracy","f1_macro","source"]
        )
        combined.to_csv(combined_path, index=False)

    # Charts
    if cv_summary is not None and not cv_summary.empty:
        save_bar(cv_summary, "model", "f1_macro_mean", "Classical 5-fold CV: macro-F1 (mean)", FIGS / "cv_f1_macro.png")
        save_bar(cv_summary, "model", "accuracy_mean", "Classical 5-fold CV: accuracy (mean)", FIGS / "cv_accuracy.png")

    if combined is not None and not combined.empty:
        # plot macro-F1 where available (drop NaN)
        avail = combined.dropna(subset=["f1_macro"]).copy()
        if not avail.empty:
            # build a label like "model (split)"
            avail["model_split"] = avail["model"].astype(str) + " [" + avail["split"].astype(str) + "]"
            save_bar(avail, "model_split", "f1_macro", "Macro-F1 available (CV means + test when present)", FIGS / "available_f1_macro.png")
        avail_acc = combined.dropna(subset=["accuracy"]).copy()
        if not avail_acc.empty:
            avail_acc["model_split"] = avail_acc["model"].astype(str) + " [" + avail_acc["split"].astype(str) + "]"
            save_bar(avail_acc, "model_split", "accuracy", "Accuracy available (CV means + test when present)", FIGS / "available_accuracy.png")

    # PDF
    c = canvas.Canvas(str(OUT_PDF), pagesize=letter)
    width, height = letter
    x = 0.75 * inch
    y = height - 0.75 * inch
    W = width - 1.5 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Clinical Note Specialty Classification — Model Report")
    y -= 22

    # Process details (specific, no fluff)
    c.setFont("Helvetica", 10)
    details = [
        f"Repo root: {ROOT}",
        f"Processed data dir: {DATA}",
        "Data split: train/val/test created by src.utils.prepare_data with train_size=0.80, val_size=0.10, test_size=0.10.",
        "Classical models trained: logistic regression, SVM, random forest, gradient boosting.",
        "Classical CV: 5-fold cross-validation (src.model.classical.cv_classical), metrics saved to data/processed/metrics_cv_*.csv.",
        "Test eval: src.model.compare_models outputs f1_macro and accuracy and writes data/processed/leaderboard.csv (if enabled).",
        "Transformer: DistilBERT fine-tuning uses Transformers v5 API (eval_strategy; Trainer processing_class).",
    ]
    for d in details:
        y = draw_wrapped(c, f"- {d}", x, y, W)
        if y < 1.0 * inch:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 0.75 * inch

    y -= 6

    # Tables
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Table 1. Classical CV summary (5-fold)")
    y -= 16
    if cv_summary is not None and not cv_summary.empty:
        show = cv_summary[["model","accuracy_mean","accuracy_std","f1_macro_mean","f1_macro_std","f1_weighted_mean","f1_weighted_std"]]
        y = add_table_text(c, show, x, y, W, max_rows=20)
    else:
        c.setFont("Helvetica", 10)
        y = draw_wrapped(c, "Missing: data/processed/metrics_cv_summary.csv", x, y, W)

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Table 2. Combined leaderboard (what is currently available)")
    y -= 16
    if combined is not None and not combined.empty:
        y = add_table_text(c, combined[["model","split","accuracy","f1_macro","source"]], x, y, W, max_rows=25)
    else:
        c.setFont("Helvetica", 10)
        y = draw_wrapped(c, "Combined leaderboard is empty.", x, y, W)

    # Figures
    figs = [
        (FIGS / "cv_f1_macro.png", "Figure 1. Classical models: CV macro-F1 (mean)."),
        (FIGS / "cv_accuracy.png", "Figure 2. Classical models: CV accuracy (mean)."),
        (FIGS / "available_f1_macro.png", "Figure 3. Macro-F1 available so far (CV + test where present)."),
        (FIGS / "available_accuracy.png", "Figure 4. Accuracy available so far (CV + test where present)."),
    ]
    for fp, cap in figs:
        if not fp.exists():
            continue
        if y < 3.0 * inch:
            c.showPage()
            y = height - 0.75 * inch
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, cap)
        y -= 12
        img = ImageReader(str(fp))
        iw, ih = img.getSize()
        max_w = W
        scale = max_w / iw
        draw_w = max_w
        draw_h = ih * scale
        if y - draw_h < 0.75 * inch:
            c.showPage()
            y = height - 0.75 * inch
        c.drawImage(img, x, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, anchor='nw')
        y -= draw_h + 18

    # Results notes (specific)
    c.showPage()
    y = height - 0.75 * inch
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Results Notes (based on saved artifacts)")
    y -= 18
    c.setFont("Helvetica", 10)

    # Pull best CV model
    if cv_summary is not None and not cv_summary.empty:
        best = cv_summary.sort_values("f1_macro_mean", ascending=False).iloc[0]
        y = draw_wrapped(c, f"- Best CV macro-F1: {best['model']} = {best['f1_macro_mean']:.6g} (± {best['f1_macro_std']:.6g}).", x, y, W)
        y = draw_wrapped(c, f"- Best CV accuracy: {cv_summary.sort_values('accuracy_mean', ascending=False).iloc[0]['model']}.", x, y, W)

    # Mention BERT completeness based on whether a local model exists
    model_dir = ROOT / "models" / "transformer" / "distilbert"
    cfg = model_dir / "config.json"
    if cfg.exists():
        y = draw_wrapped(c, f"- DistilBERT: local model artifacts found at {model_dir}.", x, y, W)
    else:
        y = draw_wrapped(c, "- DistilBERT: no config.json found under models/transformer/distilbert, so test metrics are not yet recorded in the combined table.", x, y, W)

    y -= 8
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Limitations (tight)")
    y -= 16
    c.setFont("Helvetica", 10)
    lim = [
        "Class imbalance is severe (long-tail specialties). Macro-F1 stays low even when accuracy rises.",
        "Your combined table shows NaN test metrics for several rows. Those scripts must write numeric fields to data/processed/combined_leaderboard.csv.",
        "Transformer metrics must be produced by a working evaluate module that writes results back into the same combined artifact.",
    ]
    for L in lim:
        y = draw_wrapped(c, f"- {L}", x, y, W)

    c.save()

    print("WROTE:", OUT_PDF)
    print("FIGS :", FIGS)
    print("DATA :", DATA)
    print("COMBO:", combined_path)

if __name__ == "__main__":
    main()
