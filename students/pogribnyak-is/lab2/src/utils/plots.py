import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from pathlib import Path
from typing import Optional, Sequence


def _ensure_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_conf_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    out_dir: Optional[Path] = None,
    filename: str = "confusion_matrix.png",
    labels: Optional[Sequence[int]] = (-1, 1),
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()

    if out_dir is not None:
        out_dir = _ensure_dir(Path(out_dir))
        fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    importances,
    top_k: int = 20,
    title: str = "Feature Importance (OOB^j)",
    out_dir: Optional[Path] = None,
    filename: str = "feature_importance.png",
    feature_names: Optional[Sequence[str]] = None,
):
    importances = np.array(importances)
    idx = np.argsort(importances)[-top_k:]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(idx)), importances[idx])

    if feature_names is None:
        ytick = [str(i) for i in idx]
    else:
        ytick = [feature_names[i] for i in idx]

    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(ytick)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()

    if out_dir is not None:
        out_dir = _ensure_dir(Path(out_dir))
        fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    model,
    X,
    y_true,
    title: str = "ROC Curve",
    out_dir: Optional[Path] = None,
    filename: str = "roc_curve.png",
):
    if not hasattr(model, "predict_proba"):
        print("Model has no predict_proba")
        return

    y_true = np.asarray(y_true)
    if set(np.unique(y_true)).issubset({-1, 1}):
        y_true = (y_true + 1) // 2

    y_score = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if out_dir is not None:
        out_dir = _ensure_dir(Path(out_dir))
        fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)

def plot_feature_importance_compare(custom_imp, sklearn_imp, top_k=15):
    custom_imp = np.array(custom_imp)
    sklearn_imp = np.array(sklearn_imp)

    idx = np.argsort(custom_imp)[-top_k:]

    x = np.arange(len(idx))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, custom_imp[idx], width=0.4, label="Custom")
    ax.bar(x + 0.2, sklearn_imp[idx], width=0.4, label="Sklearn")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in idx], rotation=45, ha="right")
    ax.legend()
    ax.set_title("Feature Importance Comparison")
    fig.tight_layout()
    plt.close(fig)