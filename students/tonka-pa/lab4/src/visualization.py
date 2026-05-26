from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import RocCurveDisplay, confusion_matrix  # noqa: E402

FIG_DPI = 300


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_log_likelihood_curve(history: list[float], path: Path, title: str = "Log-likelihood vs iteration") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(range(1, len(history) + 1), history, marker="o", linewidth=1.5)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("per-sample log-likelihood")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_metric_vs_k(
    k_values: list[int],
    metric_values: dict[str, list[float]],
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for label, values in metric_values.items():
        ax.plot(k_values, values, marker="o", label=label)
    ax.set_xlabel("number of components K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    _save(fig, path)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_density_2d(
    X: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str,
    sample: int = 5000,
    seed: int = 0,
) -> None:
    """Scatter of 2D embeddings colored by component assignment."""
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        idx = rng.choice(len(X), size=sample, replace=False)
        X = X[idx]
        labels = labels[idx]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    palette = sns.color_palette("tab10", n_colors=int(labels.max()) + 1)
    for k in np.unique(labels):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=6, alpha=0.5, color=palette[int(k)], label=f"comp {int(k)}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8)
    _save(fig, path)


def plot_density_contour_2d(
    model,
    X: np.ndarray,
    path: Path,
    title: str,
    grid_size: int = 120,
    sample: int = 4000,
    seed: int = 0,
) -> None:
    """Density contour over a 2D model on top of a scatter of points."""
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        idx = rng.choice(len(X), size=sample, replace=False)
        X = X[idx]
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)
    xs = np.linspace(x_min - pad_x, x_max + pad_x, grid_size)
    ys = np.linspace(y_min - pad_y, y_max + pad_y, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    log_pdf = model.score_samples(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.contour(xx, yy, log_pdf, levels=18, cmap="viridis", linewidths=0.7, alpha=0.9)
    ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.3, color="black")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    _save(fig, path)


def plot_bar_comparison(values: dict[str, float], path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    labels = list(values.keys())
    heights = list(values.values())
    bars = ax.bar(labels, heights, color=sns.color_palette("Set2", n_colors=len(labels)))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for bar, h in zip(bars, heights, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    _save(fig, path)
