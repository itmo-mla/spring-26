from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logsumexp(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable_values = np.exp(values - max_values)
    result = max_values + np.log(np.sum(stable_values, axis=axis, keepdims=True))
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


@dataclass
class _FitResult:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    lower_bound: float
    lower_bound_history: list[float]
    n_iter: int
    converged: bool


class GaussianMixtureEM:
    """Gaussian Mixture Model with full covariance matrices trained by EM."""

    def __init__(
        self,
        n_components: int = 3,
        *,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        n_init: int = 5,
        random_state: int | None = None,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if n_init < 1:
            raise ValueError("n_init must be positive")

        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "GaussianMixtureEM":
        X = self._validate_X(X)
        rng = np.random.default_rng(self.random_state)
        best_result: _FitResult | None = None

        for _ in range(self.n_init):
            result = self._fit_once(X, rng)
            if best_result is None or result.lower_bound > best_result.lower_bound:
                best_result = result

        if best_result is None:
            raise RuntimeError("GMM fitting failed")

        self.weights_ = best_result.weights
        self.means_ = best_result.means
        self.covariances_ = best_result.covariances
        self.lower_bound_ = best_result.lower_bound
        self.lower_bound_history_ = best_result.lower_bound_history
        self.n_iter_ = best_result.n_iter
        self.converged_ = best_result.converged
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_X(X)
        log_joint = self._estimate_log_prob(X) + np.log(self.weights_)
        log_norm = _logsumexp(log_joint, axis=1, keepdims=True)
        return np.exp(log_joint - log_norm)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_X(X)
        log_joint = self._estimate_log_prob(X) + np.log(self.weights_)
        return _logsumexp(log_joint, axis=1)

    def score(self, X: np.ndarray) -> float:
        """Return average log-likelihood per sample, matching sklearn's API."""
        return float(np.mean(self.score_samples(X)))

    def bic(self, X: np.ndarray) -> float:
        X = self._validate_X(X)
        n_samples, n_features = X.shape
        return -2.0 * np.sum(self.score_samples(X)) + self._n_parameters(n_features) * np.log(n_samples)

    def aic(self, X: np.ndarray) -> float:
        X = self._validate_X(X)
        n_features = X.shape[1]
        return -2.0 * np.sum(self.score_samples(X)) + 2.0 * self._n_parameters(n_features)

    def _fit_once(self, X: np.ndarray, rng: np.random.Generator) -> _FitResult:
        self._initialize_parameters(X, rng)
        history: list[float] = []
        previous_lower_bound = -np.inf
        converged = False

        for iteration in range(1, self.max_iter + 1):
            responsibilities, _ = self._e_step(X)
            self._m_step(X, responsibilities)
            lower_bound = self.score(X)
            history.append(lower_bound)

            if abs(lower_bound - previous_lower_bound) < self.tol:
                converged = True
                break
            previous_lower_bound = lower_bound

        return _FitResult(
            weights=self.weights_.copy(),
            means=self.means_.copy(),
            covariances=self.covariances_.copy(),
            lower_bound=history[-1],
            lower_bound_history=history,
            n_iter=len(history),
            converged=converged,
        )

    def _initialize_parameters(self, X: np.ndarray, rng: np.random.Generator) -> None:
        n_samples, n_features = X.shape
        indices = rng.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[indices].copy()
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

        global_covariance = np.cov(X, rowvar=False)
        global_covariance = np.atleast_2d(global_covariance)
        global_covariance.flat[:: n_features + 1] += self.reg_covar
        self.covariances_ = np.repeat(global_covariance[np.newaxis, :, :], self.n_components, axis=0)

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        log_joint = self._estimate_log_prob(X) + np.log(self.weights_)
        log_norm = _logsumexp(log_joint, axis=1, keepdims=True)
        responsibilities = np.exp(log_joint - log_norm)
        lower_bound = float(np.mean(log_norm))
        return responsibilities, lower_bound

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        n_samples, n_features = X.shape
        effective_counts = responsibilities.sum(axis=0) + 10 * np.finfo(float).eps
        self.weights_ = effective_counts / n_samples
        self.means_ = (responsibilities.T @ X) / effective_counts[:, np.newaxis]

        covariances = np.empty((self.n_components, n_features, n_features))
        for component in range(self.n_components):
            centered = X - self.means_[component]
            weighted_centered = responsibilities[:, component][:, np.newaxis] * centered
            covariance = (weighted_centered.T @ centered) / effective_counts[component]
            covariance.flat[:: n_features + 1] += self.reg_covar
            covariances[component] = covariance
        self.covariances_ = covariances

    def _estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))

        for component, (mean, covariance) in enumerate(zip(self.means_, self.covariances_)):
            try:
                cholesky = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                covariance = covariance.copy()
                covariance.flat[:: n_features + 1] += self.reg_covar
                cholesky = np.linalg.cholesky(covariance)

            centered = X - mean
            solved = np.linalg.solve(cholesky, centered.T).T
            log_det = 2.0 * np.sum(np.log(np.diag(cholesky)))
            quadratic = np.sum(solved * solved, axis=1)
            log_prob[:, component] = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + quadratic)

        return log_prob

    def _n_parameters(self, n_features: int) -> int:
        covariance_params = self.n_components * n_features * (n_features + 1) // 2
        mean_params = self.n_components * n_features
        weight_params = self.n_components - 1
        return covariance_params + mean_params + weight_params

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        return X
