# ruff: noqa: E402

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


DPI = 300


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def plot_metric_comparison(metrics: pd.DataFrame, task: str, path: Path) -> None:
    """Bar chart comparing custom vs sklearn metrics."""
    if task == "classification":
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    else:
        metric_cols = ["mse", "rmse", "mae", "r2"]
    cols = ["model", *[c for c in metric_cols if c in metrics.columns]]
    melted = metrics[cols].melt(id_vars="model", var_name="metric", value_name="value")
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=melted, x="metric", y="value", hue="model")
    plt.title("Metric comparison: custom vs sklearn")
    _finish(path)


def plot_cv_comparison(
    cv_rows: list[dict],
    task: str,
    path: Path,
) -> None:
    """Bar chart with error bars for cross-validation scores."""
    frame = pd.DataFrame(cv_rows)
    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    x = np.arange(len(frame))
    bars = ax.bar(x, frame["cv_mean"], yerr=frame["cv_std"], capsize=5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["model"], fontsize=10)
    for bar, mean in zip(bars, frame["cv_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    score_name = "accuracy" if task == "classification" else "r2"
    plt.ylabel(f"CV {score_name} (mean ± std, 5-fold)")
    plt.title("Cross-validation comparison")
    _finish(path)


def plot_stats_bar(metrics: pd.DataFrame, column: str, path: Path) -> None:
    """Bar chart for one scalar metric."""
    plt.figure(figsize=(6, 4))
    sns.barplot(data=metrics, x="model", y=column)
    plt.title(column.replace("_", " ").title())
    _finish(path)


def plot_learning_curve(
    train_loss: list[float],
    title: str,
    path: Path,
) -> None:
    """Plot training loss over boosting iterations."""
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(train_loss) + 1), train_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title(title)
    _finish(path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None,
    path: Path,
) -> None:
    """Save confusion matrix plot."""
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels)
    plt.title("Confusion matrix")
    _finish(path)


def plot_roc_curves(
    y_true: np.ndarray,
    curves: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Save ROC curves for binary classification."""
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    for name, scores in curves.items():
        RocCurveDisplay.from_predictions(y_true, scores, name=name, ax=ax)
    plt.title("ROC curve")
    _finish(path)


def plot_predicted_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Scatter predicted vs true values."""
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.45, s=15)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    _finish(path)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Scatter residuals vs predicted values."""
    residuals = np.asarray(y_true) - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.45, s=15)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(title)
    _finish(path)


def plot_feature_importances(
    feature_names: list[str],
    importances: np.ndarray,
    title: str,
    path: Path,
    top_k: int = 15,
) -> None:
    """Horizontal bar chart of top-k feature importances."""
    if importances is None or len(importances) == 0 or np.sum(importances) <= 0:
        return
    order = np.argsort(importances)[::-1][:top_k]
    frame = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "importance": importances[order],
        }
    )
    plt.figure(figsize=(8, max(4, 0.28 * len(frame))))
    sns.barplot(data=frame, y="feature", x="importance")
    plt.title(title)
    _finish(path)
