from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")
DPI = 300


def plot_class_distribution(y: pd.Series, output_path: Path) -> None:
    """Save class distribution plot."""
    counts = y.value_counts().sort_index()
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#3b82f6")
    ax.set_title("Class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_metrics_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Save a barplot comparing main quality metrics."""
    columns = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_weighted",
    ]
    available = [column for column in columns if column in metrics.columns]
    data = metrics.melt(
        id_vars="model",
        value_vars=available,
        var_name="metric",
        value_name="value",
    ).dropna()
    if data.empty:
        return

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=data, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Quality metrics comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_timing_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Save fit and predict timing comparison."""
    data = metrics.melt(
        id_vars="model",
        value_vars=["fit_time_sec", "predict_time_sec"],
        var_name="stage",
        value_name="seconds",
    )
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=data, x="stage", y="seconds", hue="model", ax=ax)
    ax.set_title("Training and prediction time")
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_oob_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Save OOB score comparison for models that expose it."""
    data = metrics[["model", "oob_score"]].dropna()
    if data.empty:
        return

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=data, x="model", y="oob_score", ax=ax, color="#10b981")
    ax.set_ylim(0, 1)
    ax.set_title("OOB score comparison")
    ax.set_xlabel("")
    ax.set_ylabel("OOB accuracy")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    """Save top impurity-based feature importances."""
    data = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    _plot_horizontal_importance(data, output_path, title)


def plot_oob_permutation_importance(
    importances: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save top OOB permutation feature importances."""
    if importances.empty:
        return
    data = importances.sort_values("importance", ascending=False).head(top_n)
    _plot_horizontal_importance(
        data[["feature", "importance"]],
        output_path,
        "OOB permutation feature importance",
    )


def plot_oob_curve(grid_results: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save OOB score versus n_estimators from grid-search results."""
    if "n_estimators" not in grid_results.columns or grid_results.empty:
        return
    data = (
        grid_results.groupby("n_estimators", as_index=False)["oob_score"]
        .max()
        .sort_values("n_estimators")
    )
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=data, x="n_estimators", y="oob_score", marker="o", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Best OOB score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_grid_search_summary(
    grid_results: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Save a heatmap or barplot summarizing grid-search scores."""
    if grid_results.empty:
        return

    _prepare_output(output_path)
    if {"max_depth", "min_samples_leaf"}.issubset(grid_results.columns):
        data = grid_results.copy()
        data["max_depth"] = data["max_depth"].fillna("None").astype(str)
        pivot = data.pivot_table(
            values="oob_score",
            index="max_depth",
            columns="min_samples_leaf",
            aggfunc="max",
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("min_samples_leaf")
        ax.set_ylabel("max_depth")
    else:
        data = grid_results.sort_values("oob_score", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=data, x=data.index.astype(str), y="oob_score", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("OOB score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def _plot_horizontal_importance(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    if data.empty:
        return

    _prepare_output(output_path)
    data = data.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.28)))
    sns.barplot(data=data, y="feature", x="importance", ax=ax, color="#6366f1")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def _prepare_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
