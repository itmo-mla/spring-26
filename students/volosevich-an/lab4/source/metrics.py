# metrics.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, silhouette_score
from scipy.special import logsumexp


# ПМП
def compute_log_likelihood(log_probs):
    return np.sum(logsumexp(log_probs, axis=1))


def bic_score(log_likelihood, n_params, n_samples):
    return -2 * log_likelihood + n_params * np.log(n_samples)


def aic_score(log_likelihood, n_params):
    return -2 * log_likelihood + 2 * n_params

# Точность через Hungarian matching
def clustering_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    row_ind, col_ind = linear_sum_assignment(-cm)

    return cm[row_ind, col_ind].sum() / np.sum(cm)


def compute_silhouette(X, labels):
    return silhouette_score(X, labels)
