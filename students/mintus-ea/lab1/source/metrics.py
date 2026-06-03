from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(name: str, model, X: pd.DataFrame, y: np.ndarray) -> dict[str, float | str]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    positive_index = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
    positive_proba = y_proba[:, positive_index]
    return {
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, positive_proba),
    }


def plot_metric_bars(metrics: pd.DataFrame, output_path) -> None:
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_data = metrics.set_index("model")[metric_columns]

    palette = ["#2F6F9F", "#56A36C", "#D89C3A", "#C7564A", "#6D5BA6"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    plot_data.plot(kind="bar", ax=ax, width=0.78, color=palette)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Classification metrics on test split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)
    ax.tick_params(axis="x", rotation=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(models: dict[str, object], X: pd.DataFrame, y: np.ndarray, output_path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        predictions = model.predict(X)
        matrix = confusion_matrix(y, predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(matrix, display_labels=["<=50K", ">50K"])
        display.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tree_complexity(before: dict[str, int], after: dict[str, int], output_path) -> None:
    labels = ["depth", "nodes", "leaves"]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - width / 2, [before[label] for label in labels], width, label="before pruning")
    ax.bar(x + width / 2, [after[label] for label in labels], width, label="after pruning")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Tree reduction effect")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, output_path, top_n: int = 10) -> None:
    top = importances.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(top.index, top.values)
    ax.set_xlabel("Normalized Gini gain")
    ax.set_title("Custom ID3 feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
