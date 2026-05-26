"""Вспомогательные функциии для графиков."""
# ruff: noqa: E402

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
from sklearn.tree import plot_tree


DPI = 300


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def plot_metric_comparison(metrics: pd.DataFrame, task: str, path: Path) -> None:
    metric_names = (
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
        if task == "classification"
        else ["mse", "rmse", "mae", "r2"]
    )
    columns = ["model", *[name for name in metric_names if name in metrics.columns]]
    melted = metrics[columns].melt(
        id_vars="model", var_name="metric", value_name="value"
    )
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=melted, x="metric", y="value", hue="model")
    plt.title("Metric comparison")
    _finish(path)


def plot_stats_bar(metrics: pd.DataFrame, column: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=metrics, x="model", y=column)
    plt.title(column.replace("_", " ").title())
    _finish(path)


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> None:
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels)
    plt.title("Confusion matrix")
    _finish(path)


def plot_roc_curves(
    y_true: np.ndarray | pd.Series,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    plt.figure(figsize=(6, 5))
    axis = plt.gca()
    for name, (scores, labels) in curves.items():
        if len(labels) == 2:
            RocCurveDisplay.from_predictions(
                y_true,
                scores,
                name=name,
                ax=axis,
                pos_label=labels[-1],
            )
    plt.title("ROC curve")
    _finish(path)


def plot_predicted_vs_true(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.55, s=18)
    lower = min(np.min(y_true), np.min(y_pred))
    upper = max(np.max(y_true), np.max(y_pred))
    plt.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    _finish(path)


def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    residuals = np.asarray(y_true) - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.55, s=18)
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
    if importances is None or len(importances) == 0 or np.sum(importances) <= 0:
        return
    order = np.argsort(importances)[::-1][:top_k]
    frame = pd.DataFrame(
        {
            "feature": [feature_names[index] for index in order],
            "importance": importances[order],
        }
    )
    plt.figure(figsize=(8, max(4, 0.28 * len(frame))))
    sns.barplot(data=frame, y="feature", x="importance")
    plt.title(title)
    _finish(path)


def plot_pruning_table(
    table: pd.DataFrame,
    score_column: str,
    title: str,
    path: Path,
) -> None:
    if table.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(table["ccp_alpha"], table["train_score"], marker="o", label="train")
    plt.plot(table["ccp_alpha"], table[score_column], marker="o", label="validation")
    plt.xlabel("ccp_alpha")
    plt.ylabel("score")
    plt.title(title)
    plt.legend()
    _finish(path)


def plot_pruning_structure(table: pd.DataFrame, title: str, path: Path) -> None:
    if table.empty:
        return
    _, axis = plt.subplots(figsize=(7, 4))
    axis.plot(table["ccp_alpha"], table["node_count"], marker="o", label="nodes")
    axis.plot(table["ccp_alpha"], table["depth"], marker="o", label="depth")
    axis.set_xlabel("ccp_alpha")
    axis.set_ylabel("count")
    axis.set_title(title)
    axis.legend()
    _finish(path)


def plot_sklearn_tree_preview(model, feature_names: list[str], path: Path) -> None:
    plt.figure(figsize=(14, 8))
    plot_tree(
        model,
        max_depth=3,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=7,
    )
    _finish(path)
