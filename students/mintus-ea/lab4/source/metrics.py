from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true: np.ndarray, labels: np.ndarray) -> float:
    """Best label matching accuracy for small number of clusters."""
    classes = np.unique(y_true)
    clusters = np.unique(labels)
    best = 0
    for permutation in itertools.permutations(classes, len(clusters)):
        mapping = {cluster: target for cluster, target in zip(clusters, permutation)}
        matched = np.array([mapping[label] for label in labels])
        best = max(best, int(np.sum(matched == y_true)))
    return best / len(y_true)


def evaluate_gmm(name: str, model, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | str | int | bool]:
    labels = model.predict(X_test)
    return {
        "model": name,
        "train_avg_log_likelihood": model.score(X_train),
        "test_avg_log_likelihood": model.score(X_test),
        "test_total_log_likelihood": float(np.sum(model.score_samples(X_test))),
        "bic_test": model.bic(X_test),
        "aic_test": model.aic(X_test),
        "clustering_accuracy": clustering_accuracy(y_test, labels),
        "adjusted_rand_index": adjusted_rand_score(y_test, labels),
        "normalized_mutual_info": normalized_mutual_info_score(y_test, labels),
        "n_iter": int(model.n_iter_),
        "converged": bool(model.converged_),
    }


def plot_log_likelihood(lower_bounds: list[float], output_path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(np.arange(1, len(lower_bounds) + 1), lower_bounds, color="#2F6F9F", linewidth=2)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Average log-likelihood")
    ax.set_title("Custom GMM EM convergence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, output_path) -> None:
    plot_data = metrics.set_index("model")[["clustering_accuracy", "adjusted_rand_index", "normalized_mutual_info"]]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    plot_data.plot(kind="bar", ax=ax, width=0.72, color=["#2F6F9F", "#56A36C", "#D89C3A"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Clustering quality on test split")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_likelihood_comparison(metrics: pd.DataFrame, output_path) -> None:
    plot_data = metrics.set_index("model")[["train_avg_log_likelihood", "test_avg_log_likelihood"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    plot_data.plot(kind="bar", ax=ax, width=0.62, color=["#6D5BA6", "#C7564A"])
    ax.set_ylabel("Average log-likelihood")
    ax.set_xlabel("")
    ax.set_title("Density estimation quality")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_projection(
    X_2d: np.ndarray,
    y_true: np.ndarray,
    custom_labels: np.ndarray,
    sklearn_labels: np.ndarray,
    output_path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    specs = [
        ("True wine class", y_true),
        ("Custom GMM clusters", custom_labels),
        ("Sklearn GMM clusters", sklearn_labels),
    ]
    for ax, (title, labels) in zip(axes, specs):
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=34, edgecolor="white", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_component_weights(custom_weights: np.ndarray, sklearn_weights: np.ndarray, output_path) -> None:
    x = np.arange(len(custom_weights))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width / 2, custom_weights, width, label="custom", color="#2F6F9F")
    ax.bar(x + width / 2, sklearn_weights, width, label="sklearn", color="#C7564A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"component {i}" for i in x])
    ax.set_ylabel("Mixture weight")
    ax.set_title("Estimated component weights")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
