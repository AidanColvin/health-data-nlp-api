"""
Generates cross-validation line graphs and significant predictor visualizations.
Extracts real features from joblib models and forces a git commit and push.
"""

import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def plot_cv_fold_accuracies(data_path: Path, out_path: Path) -> None:
    """Reads fold metrics and generates a line graph with distinct colors."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"Skipping line graph. File not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    plt.figure(figsize=(9, 6))
    
    models = df["model"].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, model in enumerate(models):
        model_data = df[df["model"] == model].sort_values("fold")
        plt.plot(
            model_data["fold"], 
            model_data["accuracy"], 
            marker=markers[i % len(markers)], 
            color=colors[i % len(colors)],
            label=model,
            linewidth=2.5
        )
        
    plt.title("Cross-Validation Accuracy by Fold Across Models")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy")
    plt.xticks(df["fold"].unique())
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Model Framework")
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved CV fold line graph to {out_path}")

def extract_and_plot_predictors(model_path: Path, out_dir: Path) -> None:
    """Extracts features from a joblib pipeline and plots top predictors."""
    if not model_path.exists():
        return

    model_name = model_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    vec, clf = None, None
    for name, step in pipeline.named_steps.items():
        if hasattr(step, 'get_feature_names_out'):
            vec = step
        if hasattr(step, 'coef_') or hasattr(step, 'feature_importances_'):
            clf = step

    if vec is None or clf is None:
        print(f"Could not find vectorizer or classifier in {model_name} pipeline.")
        return

    features = vec.get_feature_names_out()
    
    if hasattr(clf, 'coef_'):
        # For multi-class, take the maximum absolute coefficient across all classes
        importances = np.max(np.abs(clf.coef_), axis=0) if clf.coef_.ndim > 1 else clf.coef_[0]
    else:
        importances = clf.feature_importances_

    df = pd.DataFrame({"Feature": features, "Importance": importances})
    df["Abs_Importance"] = df["Importance"].abs()
    df = df.sort_values("Abs_Importance", ascending=False).head(20)
    df = df.sort_values("Abs_Importance", ascending=True) 
    
    plt.figure(figsize=(10, 8))
    plt.barh(df["Feature"], df["Abs_Importance"], color='#1f77b4')
    plt.title(f"Top 20 Significant Predictors (Magnitude): {model_name}")
    plt.xlabel("Absolute Coefficient / Feature Importance")
    plt.ylabel("Predictor (Token)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_path = out_dir / f"{model_name}_top_predictors.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved predictor graph for {model_name} to {out_path}")

def force_commit_and_push(repo_root: Path) -> None:
    """Adds visuals to git, forces a commit, and pushes to the current branch."""
    try:
        subprocess.run(["git", "add", "reports/figures/"], cwd=repo_root, check=True)
        subprocess.run(["git", "commit", "-m", "chore: auto-generate and update model visualizations"], cwd=repo_root, check=True)
        subprocess.run(["git", "push", "--force"], cwd=repo_root, check=True)
        print("Successfully committed and force-pushed visualizations.")
    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")

if __name__ == "__main__":
    repo_root = Path(".").resolve()
    
    # 1. Generate Line Graph
    cv_data_path = repo_root / "data" / "processed" / "metrics_cv_folds.csv"
    line_graph_out = repo_root / "reports" / "figures" / "cv_folds_accuracy_line.png"
    plot_cv_fold_accuracies(cv_data_path, line_graph_out)
    
    # 2. Extract and Plot Predictors for Classical Models
    predictor_out_dir = repo_root / "reports" / "figures" / "predictors"
    models_dir = repo_root / "models" / "classical"
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.joblib"):
            extract_and_plot_predictors(model_file, predictor_out_dir)
    else:
        print(f"Models directory not found at {models_dir}.")

    print("Note: BERT predictor extraction skipped. Transformer feature importance requires SHAP or LIME.")
    
    # 3. Force Git Push
    force_commit_and_push(repo_root)
