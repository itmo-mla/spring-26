import numpy as np


def _n_params(n_components: int, n_features: int) -> int:
    k, d = n_components, n_features
    return (k - 1) + k * d + k * d * (d + 1) // 2


def bic(model, X: np.ndarray) -> float:
    n = len(X)
    p = _n_params(model.n_components, X.shape[1])
    return -2 * model.score(X) * n + p * np.log(n)


def aic(model, X: np.ndarray) -> float:
    n = len(X)
    p = _n_params(model.n_components, X.shape[1])
    return -2 * model.score(X) * n + 2 * p
