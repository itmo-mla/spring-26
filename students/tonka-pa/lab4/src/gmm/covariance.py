"""
Численно устойчивые вспомогательные функции для вычисления ковариаций гауссовой смеси.

Реализует обновление логарифмической плотности для каждой выборки и обновление ковариаций в M шагов для четырёх
типов ковариаций, поддерживаемых пользовательской моделью GMM: полная, диагональная, связанная, сферическая.

Все вычисления логарифмической плотности используют разложение Холеского для ковариаций типов `full` / `tied`,
чтобы избежать явного вычисления обратной матрицы.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

LOG_2PI = float(np.log(2.0 * np.pi))


def regularize_diag(matrix: np.ndarray, reg_covar: float) -> np.ndarray:
    """Add reg_covar to the diagonal of a covariance matrix in-place-safe."""
    out = matrix.copy()
    idx = np.arange(out.shape[-1])
    out[..., idx, idx] += reg_covar
    return out


def _cholesky_safe(cov: np.ndarray, reg_covar: float) -> np.ndarray:
    """
    Cholesky factor with an automatic regularization fallback.

    Fallback предусмотрен потому, что при использовании EM может возникнуть почти сингулярная ковариация для
    компоненты с очень небольшой долей, даже если уже добавлен параметр reg_covar.
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(cov + reg_covar * np.eye(cov.shape[-1]))


def log_prob_full(X: np.ndarray, means: np.ndarray, covariances: np.ndarray, reg_covar: float) -> np.ndarray:
    """
    Per-sample log N(x; mu_k, Sigma_k) для full covariance.

    Returns array of shape (n_samples, n_components).
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components))
    const = n_features * LOG_2PI
    for k in range(n_components):
        chol = _cholesky_safe(covariances[k], reg_covar)
        # log|Sigma| = 2 * sum(log diag(L))
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        diff = X - means[k]
        # solve L y = diff^T => y = L^{-1} diff^T
        sol = solve_triangular(chol, diff.T, lower=True)
        maha = np.sum(sol**2, axis=0)
        log_prob[:, k] = -0.5 * (const + log_det + maha)
    return log_prob


def log_prob_diag(X: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """
    Per-sample log N(x; mu_k, diag(sigma2_k)).

    `covariances` has shape (n_components, n_features) and stores sigma^2 per feature.
    """
    # broadcast: (1, K, D) - (n, 1, D)
    diff = X[:, None, :] - means[None, :, :]
    inv_var = 1.0 / covariances
    # (n, K, D)
    sq = diff**2 * inv_var[None, :, :]
    log_det = np.sum(np.log(covariances), axis=1)  # (K,)
    n_features = X.shape[1]
    log_prob = -0.5 * (n_features * LOG_2PI + log_det[None, :] + sq.sum(axis=2))
    return log_prob


def log_prob_tied(X: np.ndarray, means: np.ndarray, covariance: np.ndarray, reg_covar: float) -> np.ndarray:
    """Per-sample log N(x; mu_k, Sigma) for a single tied covariance Sigma."""
    n_features = X.shape[1]
    chol = _cholesky_safe(covariance, reg_covar)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    const = n_features * LOG_2PI
    n_components = means.shape[0]
    log_prob = np.empty((X.shape[0], n_components))
    for k in range(n_components):
        diff = X - means[k]
        sol = solve_triangular(chol, diff.T, lower=True)
        maha = np.sum(sol**2, axis=0)
        log_prob[:, k] = -0.5 * (const + log_det + maha)
    return log_prob


def log_prob_spherical(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Per-sample log N(x; mu_k, sigma_k^2 * I)."""
    n_features = X.shape[1]
    diff = X[:, None, :] - means[None, :, :]
    sq = (diff**2).sum(axis=2)  # (n, K)
    inv_var = 1.0 / variances  # (K,)
    log_det = n_features * np.log(variances)  # (K,)
    log_prob = -0.5 * (n_features * LOG_2PI + log_det[None, :] + sq * inv_var[None, :])
    return log_prob


def estimate_covariances_full(
    X: np.ndarray,
    resp: np.ndarray,
    means: np.ndarray,
    nk: np.ndarray,
    reg_covar: float,
) -> np.ndarray:
    """
    M-step update for full covariance.

    Returns array of shape (n_components, n_features, n_features).
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        weighted = resp[:, k : k + 1] * diff
        cov = (weighted.T @ diff) / max(nk[k], 1e-12)
        cov.flat[:: n_features + 1] += reg_covar
        covariances[k] = cov
    return covariances


def estimate_covariances_diag(
    X: np.ndarray,
    resp: np.ndarray,
    means: np.ndarray,
    nk: np.ndarray,
    reg_covar: float,
) -> np.ndarray:
    """
    M-step update for diagonal covariance.

    Returns array of shape (n_components, n_features) — per-feature variance.
    """
    # avg X^2 weighted by resp
    avg_X2 = (resp.T @ (X**2)) / np.clip(nk[:, None], 1e-12, None)
    avg_means2 = means**2
    avg_X_means = means * ((resp.T @ X) / np.clip(nk[:, None], 1e-12, None))
    variances = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
    return np.clip(variances, reg_covar, None)


def estimate_covariances_tied(
    X: np.ndarray,
    resp: np.ndarray,
    means: np.ndarray,
    nk: np.ndarray,
    reg_covar: float,
) -> np.ndarray:
    """
    M-step update for tied covariance (shared across components).

    Returns array of shape (n_features, n_features).
    """
    n_samples, n_features = X.shape
    # Sigma = (sum_k sum_i resp_{i,k} (x_i - mu_k)(x_i - mu_k)^T) / N
    cov = np.zeros((n_features, n_features))
    for k in range(means.shape[0]):
        diff = X - means[k]
        weighted = resp[:, k : k + 1] * diff
        cov += weighted.T @ diff
    cov /= n_samples
    cov.flat[:: n_features + 1] += reg_covar
    return cov


def estimate_covariances_spherical(
    X: np.ndarray,
    resp: np.ndarray,
    means: np.ndarray,
    nk: np.ndarray,
    reg_covar: float,
) -> np.ndarray:
    """
    M-step update for spherical covariance.

    Returns array of shape (n_components,) — per-component scalar variance.
    """
    diag_var = estimate_covariances_diag(X, resp, means, nk, reg_covar)
    return diag_var.mean(axis=1)


def n_parameters(covariance_type: str, n_components: int, n_features: int) -> int:
    """
    Number of free parameters for BIC/AIC.

    Matches the convention used by sklearn's GaussianMixture.
    """
    K, D = n_components, n_features
    means_params = K * D
    mix_params = K - 1
    if covariance_type == "full":
        cov_params = K * D * (D + 1) // 2
    elif covariance_type == "diag":
        cov_params = K * D
    elif covariance_type == "tied":
        cov_params = D * (D + 1) // 2
    elif covariance_type == "spherical":
        cov_params = K
    else:
        raise ValueError(f"unknown covariance_type: {covariance_type}")
    return int(mix_params + means_params + cov_params)
