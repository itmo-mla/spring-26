from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp


class GaussianMixtureModel:
    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.lower_bound_: float = np.nan
        self.converged_: bool = False

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self._initialize(X, rng)

        prev_lower = -np.inf
        for iteration in range(self.max_iter):
            log_resp, log_prob_norm = self._e_step(X)
            self._m_step(X, np.exp(log_resp))
            lower = float(np.sum(log_prob_norm))
            self.lower_bound_ = lower

            if iteration > 0 and abs(lower - prev_lower) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                break
            prev_lower = lower
            self.n_iter_ = iteration + 1
        else:
            self.converged_ = False

        return self

    def _initialize(self, X: np.ndarray, rng: np.random.Generator) -> None:
        n_samples, n_features = X.shape
        k = self.n_components

        indices = rng.choice(n_samples, size=k, replace=False)
        self.means_ = X[indices].copy()
        self.covariances_ = np.array(
            [np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features) for _ in range(k)]
        )
        self.weights_ = np.full(k, 1.0 / k)

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_prob = self._estimate_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-300)
        log_prob_weighted = log_prob + log_weights

        log_prob_norm = logsumexp(log_prob_weighted, axis=1)
        log_resp = log_prob_weighted - log_prob_norm[:, np.newaxis]
        return log_resp, log_prob_norm

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        n_samples, n_features = X.shape
        k = self.n_components

        nk = resp.sum(axis=0) + 1e-300
        self.weights_ = nk / n_samples
        self.means_ = (resp.T @ X) / nk[:, np.newaxis]

        covariances = np.empty((k, n_features, n_features))
        for component in range(k):
            diff = X - self.means_[component]
            covariances[component] = (resp[:, component] * diff.T) @ diff / nk[component]
            covariances[component].flat[:: n_features + 1] += self.reg_covar
        self.covariances_ = covariances

    def _estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        log_prob = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = self._log_gaussian_pdf(X, self.means_[k], self.covariances_[k])
        return log_prob

    @staticmethod
    def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        chol = cholesky(cov, lower=True)
        diff = X - mean
        solved = solve_triangular(chol, diff.T, lower=True)
        maha_sq = np.sum(solved ** 2, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + maha_sq)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        log_prob = self._estimate_log_prob(np.asarray(X, dtype=np.float64))
        log_weights = np.log(self.weights_ + 1e-300)
        return logsumexp(log_prob + log_weights, axis=1)

    def score(self, X: np.ndarray) -> float:
        return float(np.mean(self.score_samples(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_resp, _ = self._e_step(np.asarray(X, dtype=np.float64))
        return np.argmax(log_resp, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_resp, _ = self._e_step(np.asarray(X, dtype=np.float64))
        return np.exp(log_resp)
