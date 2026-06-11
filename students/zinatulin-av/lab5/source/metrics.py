import numpy as np

def calculate_rmse(R_true, R_pred):
    R_true = np.asarray(R_true, dtype=float)
    R_pred = np.asarray(R_pred, dtype=float)
    return np.sqrt(np.mean((R_true - R_pred) ** 2))

def _dcg(relevance, k):
    relevance = relevance[:k]
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return np.sum(relevance / discounts)

def _ndcg_single(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    ranked_true = y_true[order]
    dcg = _dcg(ranked_true, k)
    ideal = _dcg(np.sort(y_true)[::-1], k)
    if ideal == 0:
        return 0.0
    return dcg / ideal

def calculate_ndcg(R_true, R_pred, k):
    R_true = np.asarray(R_true, dtype=float)
    R_pred = np.asarray(R_pred, dtype=float)
    ndcg_vals = []
    for i in range(R_true.shape[0]):
        if np.sum(R_true[i]) > 0:
            ndcg_vals.append(_ndcg_single(R_true[i], R_pred[i], k))
    return np.mean(ndcg_vals)
