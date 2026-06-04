import numpy as np
from scipy import sparse


def rmse_on_entries(predictions: np.ndarray, rows: np.ndarray, columns: np.ndarray, values: np.ndarray) -> float:
    predicted_values = predictions[rows, columns]
    return float(np.sqrt(np.mean((predicted_values - values) ** 2)))


def ndcg_at_k(
    predictions: np.ndarray,
    train_matrix: sparse.spmatrix,
    holdout_rows: np.ndarray,
    holdout_columns: np.ndarray,
    holdout_values: np.ndarray,
    k: int = 10,
) -> float:
    train_csr = train_matrix.tocsr()
    holdout_by_row: dict[int, list[tuple[int, float]]] = {}
    for row, column, value in zip(holdout_rows, holdout_columns, holdout_values, strict=True):
        holdout_by_row.setdefault(int(row), []).append((int(column), float(value)))

    scores: list[float] = []
    for row, relevant_items in holdout_by_row.items():
        ranking_scores = predictions[row].copy()
        ranking_scores[train_csr[row].indices] = -np.inf

        top_k = min(k, ranking_scores.size)
        top_indices = np.argpartition(ranking_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(ranking_scores[top_indices])[::-1]]

        relevance = {column: value for column, value in relevant_items}
        dcg = 0.0
        for rank, column in enumerate(top_indices[:k], start=1):
            gain = relevance.get(int(column), 0.0)
            dcg += gain / np.log2(rank + 1)

        ideal_values = sorted((value for _, value in relevant_items), reverse=True)[:k]
        ideal_dcg = sum(value / np.log2(rank + 1) for rank, value in enumerate(ideal_values, start=1))
        if ideal_dcg > 0.0:
            scores.append(dcg / ideal_dcg)

    return float(np.mean(scores)) if scores else 0.0
