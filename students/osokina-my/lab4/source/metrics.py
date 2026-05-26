import numpy as np


def log_likelihood(log_probs: np.ndarray) -> float:
    return float(np.sum(log_probs))


def average_log_likelihood(log_probs: np.ndarray) -> float:
    return float(np.mean(log_probs))


def aic(log_likelihood_sum: float, n_params: int) -> float:
    return 2 * n_params - 2 * log_likelihood_sum


def bic(log_likelihood_sum: float, n_params: int, n_samples: int) -> float:
    return n_params * np.log(n_samples) - 2 * log_likelihood_sum


def count_gmm_params(n_features: int, n_components: int) -> int:
    per_component = n_features + n_features * (n_features + 1) // 2
    return (n_components - 1) + n_components * per_component
