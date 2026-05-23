import logging
import os
import urllib.request
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def load_data(test_size: float = 0.2, random_state: int = 42):
    cache_dir = os.path.join(os.path.expanduser("~"), ".ml100k_cache")
    ratings_path = os.path.join(cache_dir, "ml-100k", "u.data")

    if not os.path.exists(ratings_path):
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Скачиваю MovieLens 100K из {DATASET_URL}...")
        zip_path = os.path.join(cache_dir, "ml-100k.zip")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
        os.remove(zip_path)

    triples = []
    with open(ratings_path, "r") as f:
        for line in f:
            u, i, r, _ = line.strip().split("\t")
            triples.append((int(u) - 1, int(i) - 1, float(r)))

    triples = np.array(triples)
    train_idx, test_idx = train_test_split(
        np.arange(len(triples)), test_size=test_size, random_state=random_state
    )

    n_users = int(triples[:, 0].max()) + 1
    n_items = int(triples[:, 1].max()) + 1

    R_train = np.zeros((n_users, n_items))
    R_test = np.zeros((n_users, n_items))
    for u, i, r in triples[train_idx]:
        R_train[int(u), int(i)] = r
    for u, i, r in triples[test_idx]:
        R_test[int(u), int(i)] = r

    return R_train, R_test, R_train > 0, R_test > 0


def compute_rmse(R_true: np.ndarray, R_pred: np.ndarray, mask: np.ndarray) -> float:
    diff = (R_pred - R_true)[mask]
    return float(np.sqrt((diff ** 2).mean()))


def ndcg_at_k(
    R_true: np.ndarray,
    scores: np.ndarray,
    mask_train: np.ndarray,
    k: int = 10,
) -> float:
    n_users = R_true.shape[0]
    log_pos = np.log2(np.arange(k) + 2)
    vals = []

    for u in range(n_users):
        true_row = R_true[u]
        if not true_row.any():
            continue

        # убираем из ранжирования предметы из обучающей выборки
        s = scores[u].copy()

        # не рекомендуем уже оценённые
        s[mask_train[u]] = -np.inf
        top = np.argsort(-s)[:k]
        dcg = (true_row[top] / log_pos).sum()

        ideal = np.sort(true_row)[::-1][:k]
        idcg = (ideal / log_pos).sum()

        if idcg > 0:
            vals.append(dcg / idcg)

    return float(np.mean(vals)) if vals else 0.0


def save_metrics(path: str, label: str, rmse: float, ndcg: float, time_s: float) -> dict:
    report = (
        f"{label}\n"
        f"RMSE: {rmse:.4f}\n"
        f"NDCG@10: {ndcg:.4f}\n"
        f"Время обучения: {time_s:.2f}s\n"
    )
    logging.info("\n" + report)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return {"rmse": rmse, "ndcg": ndcg, "time": time_s}


def plot_comparison(metrics: dict, save_path: str):
    labels = list(metrics.keys())
    rmse_vals = [metrics[l]["rmse"] for l in labels]
    ndcg_vals = [metrics[l]["ndcg"] for l in labels]

    x = np.arange(len(labels))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, vals, ylabel, title in (
        (axes[0], rmse_vals, "RMSE", "RMSE (меньше — лучше)"),
        (axes[1], ndcg_vals, "NDCG@10", "NDCG@10 (больше — лучше)"),
    ):
        ax.bar(x, vals, color=colors[: len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1.0)
        for xi, v in zip(x, vals):
            ax.text(xi, v + max(vals) * 0.01, f"{v:.4f}", ha="center", fontsize=9)

    plt.suptitle("Сравнение моделей рекомендательных систем", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
