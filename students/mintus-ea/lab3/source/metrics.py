from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(name: str, model, X: pd.DataFrame, y: np.ndarray, train_time: float) -> dict[str, float | str]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "log_loss": log_loss(y, np.column_stack([1.0 - probabilities, probabilities])),
        "train_time_sec": train_time,
    }


def summarize_cv(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "train_time_sec"]
    summary_rows = []
    for model_name, group in frame.groupby("model", sort=False):
        row = {"model": model_name}
        for column in metric_columns:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_std"] = group[column].std(ddof=0)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def plot_cv_metrics(summary: pd.DataFrame, output_path, cv_folds: int) -> None:
    metric_names = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
    plot_data = summary.set_index("model")[metric_names]
    plot_data.columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    plot_data.plot(kind="bar", ax=ax, width=0.76, color=["#2F6F9F", "#56A36C", "#D89C3A", "#C7564A", "#6D5BA6"])
    ax.set_ylim(0.88, 1.01)
    ax.set_ylabel("CV mean score")
    ax.set_xlabel("")
    ax.set_title(f"{cv_folds}-fold cross-validation quality")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_time(cv_summary: pd.DataFrame, test_metrics: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    labels = []
    values = []
    for _, row in cv_summary.iterrows():
        labels.append(f"{row['model']} CV fold")
        values.append(row["train_time_sec_mean"])
    for _, row in test_metrics.iterrows():
        labels.append(f"{row['model']} final")
        values.append(row["train_time_sec"])

    ax.bar(labels, values, color=["#2F6F9F", "#56A36C", "#7C6BB0", "#C7564A"])
    ax.set_ylabel("Seconds")
    ax.set_title("Training time")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_learning_curve(loss_values: list[float], output_path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(np.arange(1, len(loss_values) + 1), loss_values, color="#2F6F9F", linewidth=2)
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Train log loss")
    ax.set_title("Custom Gradient Boosting optimization trace")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(models: dict[str, object], X: pd.DataFrame, y: np.ndarray, output_path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        matrix = confusion_matrix(y, model.predict(X), labels=[0, 1])
        display = ConfusionMatrixDisplay(matrix, display_labels=["benign", "malignant"])
        display.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importances(
    feature_names: list[str],
    custom_importance: np.ndarray,
    sklearn_importance: np.ndarray,
    output_path,
    top_n: int = 12,
) -> None:
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "custom": custom_importance,
            "sklearn": sklearn_importance,
        }
    )
    top_features = (
        frame.assign(total=frame["custom"] + frame["sklearn"])
        .sort_values("total", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    y_pos = np.arange(len(top_features))
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.barh(y_pos - 0.18, top_features["custom"], height=0.36, label="custom", color="#2F6F9F")
    ax.barh(y_pos + 0.18, top_features["sklearn"], height=0.36, label="sklearn", color="#C7564A")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Normalized importance")
    ax.set_title("Gradient Boosting feature importances")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
