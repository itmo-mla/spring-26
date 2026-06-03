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
)


def evaluate_model(name: str, model, X: pd.DataFrame, y: np.ndarray, train_time: float) -> dict[str, float | str]:
    predictions = model.predict(X)
    return {
        "model": name,
        "accuracy": accuracy_score(y, predictions),
        "precision_macro": precision_score(y, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y, predictions, average="macro", zero_division=0),
        "train_time_sec": train_time,
        "oob_score": getattr(model, "oob_score_", np.nan),
    }


def plot_metrics(metrics: pd.DataFrame, output_path) -> None:
    plot_data = metrics.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro", "oob_score"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    plot_data.plot(kind="bar", ax=ax, width=0.78, color=["#2F6F9F", "#56A36C", "#D89C3A", "#C7564A", "#6D5BA6"])
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Random Forest metrics on Digits test split")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_time(metrics: pd.DataFrame, grid_time: float, output_path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    labels = list(metrics["model"]) + ["Custom GridSearchCV"]
    values = list(metrics["train_time_sec"]) + [grid_time]
    ax.bar(labels, values, color=["#2F6F9F", "#56A36C", "#8C6BB1"])
    ax.set_ylabel("Seconds")
    ax.set_title("Training time comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(models: dict[str, object], X: pd.DataFrame, y: np.ndarray, output_path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.2))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        matrix = confusion_matrix(y, model.predict(X), labels=np.arange(10))
        display = ConfusionMatrixDisplay(matrix, display_labels=np.arange(10))
        display.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_grid_search(grid_results: pd.DataFrame, output_path, top_n: int = 10) -> None:
    top = grid_results.sort_values("mean_test_score", ascending=False).head(top_n).iloc[::-1]
    labels = [
        f"n={row['param_n_estimators']}, depth={row['param_max_depth']}, "
        f"feat={row['param_max_features']}, leaf={row['param_min_samples_leaf']}"
        for _, row in top.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(labels, top["mean_test_score"], color="#3D7C7A")
    ax.set_xlim(max(0.85, top["mean_test_score"].min() - 0.01), min(1.0, top["mean_test_score"].max() + 0.01))
    ax.set_xlabel("OOB accuracy")
    ax.set_title("Top GridSearchCV configurations by OOB")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_importance_heatmaps(
    custom_oob_importance: pd.DataFrame,
    sklearn_importance: np.ndarray,
    output_path,
) -> None:
    custom_grid = np.zeros(64, dtype=float)
    for _, row in custom_oob_importance.iterrows():
        _, row_id, col_id = str(row["feature"]).split("_")
        index = int(row_id) * 8 + int(col_id)
        custom_grid[index] = row["importance_normalized"]

    sklearn_grid = sklearn_importance / sklearn_importance.sum()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, grid, title in [
        (axes[0], custom_grid.reshape(8, 8), "Custom RF OOB^j importance"),
        (axes[1], sklearn_grid.reshape(8, 8), "Sklearn RF impurity importance"),
    ]:
        image = ax.imshow(grid, cmap="magma")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_importances(custom_oob_importance: pd.DataFrame, output_path, top_n: int = 12) -> None:
    top = custom_oob_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4F7EA8")
    ax.set_xlabel("OOB accuracy decrease")
    ax.set_title("Top OOB^j feature importances")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
