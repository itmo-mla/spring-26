from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import ndcg_score, root_mean_squared_error


def rating_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def ranking_ndcg(
    ground_truth: sparse.csr_matrix,
    predictions: np.ndarray,
    *,
    k: int = 10,
) -> float:
    scores: list[float] = []

    for user_idx in range(ground_truth.shape[0]):
        y_true = ground_truth.getrow(user_idx).toarray()
        if y_true.sum() == 0:
            continue
        y_score = predictions[user_idx].reshape(1, -1)
        scores.append(float(ndcg_score(y_true, y_score, k=k)))

    return float(np.mean(scores)) if scores else 0.0
