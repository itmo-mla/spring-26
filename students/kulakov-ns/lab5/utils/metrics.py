from typing import Dict

import numpy as np
import pandas as pd


def rmse_on_interactions(prediction_matrix: np.ndarray, interactions: pd.DataFrame) -> float:
    user_indices = interactions["user_index"].to_numpy(dtype=int)
    item_indices = interactions["item_index"].to_numpy(dtype=int)
    true_ratings = interactions["rating"].to_numpy(dtype=float)
    predicted_ratings = prediction_matrix[user_indices, item_indices]
    return float(np.sqrt(np.mean((true_ratings - predicted_ratings) ** 2)))


def ndcg_at_k(
    prediction_matrix: np.ndarray,
    test_interactions: pd.DataFrame,
    train_matrix: np.ndarray,
    k: int = 10,
) -> float:
    scores = []
    n_users, n_items = prediction_matrix.shape

    for user_index in range(n_users):
        user_test = test_interactions[test_interactions["user_index"] == user_index]
        if user_test.empty:
            continue

        relevance = np.zeros(n_items, dtype=float)
        for row in user_test.itertuples(index=False):
            relevance[int(row.item_index)] = float(row.rating)

        user_scores = prediction_matrix[user_index].copy()
        user_scores[train_matrix[user_index] > 0] = -np.inf

        top_k = np.argsort(user_scores)[::-1][:k]
        dcg = 0.0
        for rank, item_index in enumerate(top_k, start=1):
            rel = relevance[item_index]
            dcg += (2.0 ** rel - 1.0) / np.log2(rank + 1.0)

        ideal_relevance = np.sort(user_test["rating"].to_numpy(dtype=float))[::-1][:k]
        idcg = 0.0
        for rank, rel in enumerate(ideal_relevance, start=1):
            idcg += (2.0 ** rel - 1.0) / np.log2(rank + 1.0)

        if idcg > 0:
            scores.append(dcg / idcg)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def evaluate_model(
    name: str,
    model,
    train_matrix: np.ndarray,
    test_interactions: pd.DataFrame,
    k: int = 10,
) -> Dict[str, float]:
    prediction_matrix = model.predict_matrix(train_matrix)
    prediction_matrix = np.clip(prediction_matrix, 1.0, 5.0)
    return {
        "model": name,
        "rmse": rmse_on_interactions(prediction_matrix, test_interactions),
        f"ndcg@{k}": ndcg_at_k(prediction_matrix, test_interactions, train_matrix, k=k),
    }
