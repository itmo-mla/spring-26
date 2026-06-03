import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ndcg_at_k(rows, y_true, y_score, k=10):
    rows = np.asarray(rows)
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    values = []

    for user in np.unique(rows):
        mask = rows == user
        true_user = y_true[mask]
        score_user = y_score[mask]

        if true_user.size == 0 or np.sum(true_user) == 0:
            continue

        order = np.argsort(score_user)[::-1][:k]
        ideal = np.argsort(true_user)[::-1][:k]

        gains = true_user[order]
        ideal_gains = true_user[ideal]
        discounts = 1 / np.log2(np.arange(2, gains.size + 2))
        ideal_discounts = 1 / np.log2(np.arange(2, ideal_gains.size + 2))

        dcg = np.sum(gains * discounts)
        idcg = np.sum(ideal_gains * ideal_discounts)

        if idcg > 0:
            values.append(dcg / idcg)

    if not values:
        return 0.0

    return float(np.mean(values))
