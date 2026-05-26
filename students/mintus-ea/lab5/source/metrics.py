from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmse_on_entries(prediction: np.ndarray, entries: pd.DataFrame) -> float:
    y_true = entries["rating"].to_numpy(dtype=float)
    y_pred = prediction[entries["user"].to_numpy(dtype=int), entries["item"].to_numpy(dtype=int)]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ndcg_at_k(prediction: np.ndarray, train_matrix: np.ndarray, entries: pd.DataFrame, k: int = 10) -> float:
    grouped = entries.groupby("user")
    scores = []
    n_items = prediction.shape[1]

    for user_id, group in grouped:
        user_id = int(user_id)
        relevance = dict(zip(group["item"].astype(int), group["rating"].astype(float)))
        candidate_scores = prediction[user_id].copy()
        candidate_scores[train_matrix[user_id] > 0] = -np.inf
        if np.all(~np.isfinite(candidate_scores)):
            continue

        top_items = np.argsort(candidate_scores)[::-1][: min(k, n_items)]
        dcg = 0.0
        for rank, item_id in enumerate(top_items, start=1):
            rel = relevance.get(int(item_id), 0.0)
            dcg += rel / np.log2(rank + 1)

        ideal_rels = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
        if idcg > 0:
            scores.append(dcg / idcg)

    return float(np.mean(scores)) if scores else 0.0


def evaluate_model(
    name: str,
    family: str,
    prediction: np.ndarray,
    train_matrix: np.ndarray,
    test_entries: pd.DataFrame,
    train_time: float,
    k: int = 10,
) -> dict[str, float | str]:
    return {
        "model": name,
        "family": family,
        "rmse": rmse_on_entries(prediction, test_entries),
        f"ndcg@{k}": ndcg_at_k(prediction, train_matrix, test_entries, k=k),
        "train_time_sec": train_time,
    }


def top_recommendations(
    prediction: np.ndarray,
    train_matrix: np.ndarray,
    feature_names: list[str],
    topics: list[str],
    users: list[int],
    model_name: str,
    top_k: int = 8,
) -> pd.DataFrame:
    rows = []
    for user_id in users:
        scores = prediction[user_id].copy()
        scores[train_matrix[user_id] > 0] = -np.inf
        top_items = np.argsort(scores)[::-1][:top_k]
        for rank, item_id in enumerate(top_items, start=1):
            rows.append(
                {
                    "model": model_name,
                    "document": user_id,
                    "topic": topics[user_id],
                    "rank": rank,
                    "term": feature_names[item_id],
                    "score": float(prediction[user_id, item_id]),
                }
            )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, output_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    ordered = metrics.sort_values(["family", "model"])
    axes[0].bar(ordered["model"], ordered["rmse"], color="#2F6F9F")
    axes[0].set_title("RMSE on held-out TF-IDF entries")
    axes[0].set_ylabel("RMSE, lower is better")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(axis="y", alpha=0.25)

    ndcg_column = [column for column in ordered.columns if column.startswith("ndcg@")][0]
    axes[1].bar(ordered["model"], ordered[ndcg_column], color="#56A36C")
    axes[1].set_title(f"{ndcg_column.upper()} for held-out terms")
    axes[1].set_ylabel("NDCG, higher is better")
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_time(metrics: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ordered = metrics.sort_values("train_time_sec")
    ax.bar(ordered["model"], ordered["train_time_sec"], color=["#56A36C", "#D89C3A", "#2F6F9F", "#C7564A"])
    ax.set_ylabel("Seconds")
    ax.set_title("Training time")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_singular_values(values_by_model: dict[str, np.ndarray], output_path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for name, values in values_by_model.items():
        ax.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=2, label=name)
    ax.set_xlabel("Latent component")
    ax.set_ylabel("Singular value")
    ax.set_title("Latent semantic spectrum")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_matrix_density(original_matrix: np.ndarray, train_matrix: np.ndarray, output_path) -> None:
    values = [
        np.count_nonzero(original_matrix) / original_matrix.size,
        np.count_nonzero(train_matrix) / train_matrix.size,
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(["full TF-IDF", "train after holdout"], values, color=["#6D5BA6", "#2F6F9F"])
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel("Density")
    ax.set_title("Document-term matrix density")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
