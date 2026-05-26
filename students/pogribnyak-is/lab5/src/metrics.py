import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_from_df(df: pd.DataFrame, pred_matrix: np.ndarray,
                 u2i: dict, v2i: dict) -> float:
    y_true, y_pred = [], []
    for row in df.itertuples(index=False):
        u = u2i.get(row.user_id)
        v = v2i.get(row.item_id)
        if u is None or v is None:
            continue
        y_true.append(row.rating)
        y_pred.append(pred_matrix[u, v])
    return rmse(np.array(y_true), np.array(y_pred))


def dcg_at_k(scores: np.ndarray, k: int) -> float:
    scores = scores[:k]
    return float(np.sum(scores / np.log2(np.arange(2, len(scores) + 2))))


def ndcg_at_k(pred_matrix: np.ndarray, R_true: np.ndarray, k: int = 10) -> float:
    ndcgs = []
    for u in range(pred_matrix.shape[0]):
        true_ratings = R_true[u]
        if true_ratings.sum() == 0:
            continue
        ranked_idx = np.argsort(pred_matrix[u])[::-1]
        ideal_idx = np.argsort(true_ratings)[::-1]

        rel_ranked = true_ratings[ranked_idx]
        rel_ideal = true_ratings[ideal_idx]

        actual_dcg = dcg_at_k(rel_ranked, k)
        ideal_dcg = dcg_at_k(rel_ideal, k)
        if ideal_dcg > 0:
            ndcgs.append(actual_dcg / ideal_dcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
