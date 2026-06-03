from typing import Any, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score



def clustering_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    size = max(len(true_labels), len(pred_labels))
    confusion = np.zeros((size, size), dtype=int)

    true_index = {label: index for index, label in enumerate(true_labels)}
    pred_index = {label: index for index, label in enumerate(pred_labels)}

    for true_value, pred_value in zip(y_true, y_pred):
        confusion[pred_index[pred_value], true_index[true_value]] += 1

    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    matched = confusion[row_ind, col_ind].sum()
    return float(matched / len(y_true))



def evaluate_model(name, model, X, y) -> Dict[str, Any]:
    pred = model.predict(X)
    return {
        "model": name,
        "avg_log_likelihood": float(model.score(X)),
        "accuracy": clustering_accuracy(y, pred),
        "ari": adjusted_rand_score(y, pred),
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
    }
