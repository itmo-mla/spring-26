import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from pathlib import Path

PLOTS_DIR = Path("plots")
COLORS = list(mcolors.TABLEAU_COLORS.values())


def _save(fig: plt.Figure, name: str):
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ellipse(ax, mean, cov, n_std: float = 2.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h = 2 * n_std * np.sqrt(vals[::-1])
    ax.add_patch(Ellipse(mean, w, h, angle=angle, **kw))


def _ax_labels(ax):
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")


def plot_data(X: np.ndarray, dataset_name: str = "Dataset"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], s=30, alpha=0.6, color="#4C72B0")
    _ax_labels(ax)
    ax.set_title(f"{dataset_name} — raw data")
    _save(fig, "data_distribution.png")


def plot_gmm(X: np.ndarray, model, title: str, filename: str):
    labels = model.predict(X)
    fig, ax = plt.subplots(figsize=(8, 6))

    for k in range(model.n_components):
        c = COLORS[k % len(COLORS)]
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=25, alpha=0.6, color=c, label=f"#{k+1}")
        _ellipse(ax, model.means_[k], model.covariances_[k],
                 facecolor="none", edgecolor=c, linewidth=2)
        ax.scatter(*model.means_[k], marker="x", s=90, linewidths=2, color=c, zorder=5)

    _ax_labels(ax)
    ax.set_title(title)
    ax.legend(title="Component", fontsize=9)
    _save(fig, filename)


def plot_convergence(log_likelihoods: list[float]):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_likelihoods, color="#2196F3", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Log-Likelihood / sample")
    ax.set_title("EM Algorithm — Convergence")
    ax.grid(alpha=0.3)
    _save(fig, "convergence.png")


def plot_model_selection(ks: list[int], bics: list[float], aics: list[float]):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, bics, "o-", label="BIC", color="#E74C3C", linewidth=2)
    ax.plot(ks, aics, "s--", label="AIC", color="#2ECC71", linewidth=2)
    best = ks[int(np.argmin(bics))]
    ax.axvline(best, color="gray", linestyle=":", alpha=0.7, label=f"Best k={best}")
    ax.set_xlabel("Number of Components (k)")
    ax.set_ylabel("Criterion Value (lower = better)")
    ax.set_title("Model Selection via BIC / AIC")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "model_selection.png")
