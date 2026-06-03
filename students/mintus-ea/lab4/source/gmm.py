from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class GaussianMixtureEM(BaseEstimator, ClusterMixin):
    """Full-covariance Gaussian Mixture Model fitted by the EM algorithm."""

    def __init__(
        self,
        n_components: int = 3,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        init_params: str = "kmeans",
        n_init: int = 3,
        random_state: int | None = 42,
    ) -> None:
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None) -> "GaussianMixtureEM":
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        best_state = None
        best_bound = -np.inf

        for _ in range(self.n_init):
            seed = int(rng.randint(0, np.iinfo(np.int32).max))
            state = self._fit_once(X, seed)
            if state["lower_bound"] > best_bound:
                best_bound = state["lower_bound"]
                best_state = state

        self.weights_ = best_state["weights"]
        self.means_ = best_state["means"]
        self.covariances_ = best_state["covariances"]
        self.converged_ = best_state["converged"]
        self.n_iter_ = best_state["n_iter"]
        self.lower_bound_ = best_state["lower_bound"]
        self.lower_bounds_ = best_state["lower_bounds"]
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        return self._logsumexp(weighted_log_prob, axis=1)

    def score(self, X, y=None) -> float:
        return float(np.mean(self.score_samples(X)))

    def bic(self, X) -> float:
        X = np.asarray(X, dtype=float)
        return -2.0 * np.sum(self.score_samples(X)) + self._n_parameters() * np.log(len(X))

    def aic(self, X) -> float:
        X = np.asarray(X, dtype=float)
        return -2.0 * np.sum(self.score_samples(X)) + 2.0 * self._n_parameters()

    def _fit_once(self, X: np.ndarray, seed: int) -> dict[str, object]:
        rng = np.random.RandomState(seed)
        weights, means, covariances = self._initialize_parameters(X, rng)
        lower_bounds: list[float] = []
        converged = False
        previous_bound = -np.inf

        for iteration in range(1, self.max_iter + 1):
            log_prob_norm, log_resp = self._estimate_log_prob_resp_with_params(
                X,
                weights,
                means,
                covariances,
            )
            responsibilities = np.exp(log_resp)
            weights, means, covariances = self._m_step(X, responsibilities)

            lower_bound = float(np.mean(log_prob_norm))
            lower_bounds.append(lower_bound)
            if abs(lower_bound - previous_bound) < self.tol:
                converged = True
                break
            previous_bound = lower_bound

        return {
            "weights": weights,
            "means": means,
            "covariances": covariances,
            "converged": converged,
            "n_iter": iteration,
            "lower_bound": lower_bounds[-1],
            "lower_bounds": lower_bounds,
        }

    def _initialize_parameters(self, X: np.ndarray, rng: np.random.RandomState):
        if self.init_params == "random":
            labels = rng.randint(self.n_components, size=len(X))
            means = np.vstack([X[labels == k].mean(axis=0) if np.any(labels == k) else X[rng.randint(len(X))] for k in range(self.n_components)])
        elif self.init_params == "kmeans":
            means, labels = self._kmeans_init(X, rng)
        else:
            raise ValueError("init_params must be either 'kmeans' or 'random'.")

        weights = np.zeros(self.n_components, dtype=float)
        covariances = np.zeros((self.n_components, X.shape[1], X.shape[1]), dtype=float)
        global_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(X.shape[1])

        for component in range(self.n_components):
            mask = labels == component
            if np.any(mask):
                weights[component] = np.mean(mask)
                centered = X[mask] - means[component]
                covariances[component] = centered.T @ centered / max(1, mask.sum())
            else:
                weights[component] = 1.0 / self.n_components
                covariances[component] = global_cov.copy()
            covariances[component].flat[:: X.shape[1] + 1] += self.reg_covar

        weights = weights / weights.sum()
        return weights, means, covariances

    def _kmeans_init(self, X: np.ndarray, rng: np.random.RandomState):
        n_samples, n_features = X.shape
        centers = np.empty((self.n_components, n_features), dtype=float)
        centers[0] = X[rng.randint(n_samples)]
        closest_sq_dist = np.sum((X - centers[0]) ** 2, axis=1)

        for component in range(1, self.n_components):
            total = closest_sq_dist.sum()
            if total <= 0:
                centers[component] = X[rng.randint(n_samples)]
            else:
                centers[component] = X[rng.choice(n_samples, p=closest_sq_dist / total)]
            closest_sq_dist = np.minimum(closest_sq_dist, np.sum((X - centers[component]) ** 2, axis=1))

        labels = np.zeros(n_samples, dtype=int)
        for _ in range(30):
            distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for component in range(self.n_components):
                if np.any(labels == component):
                    centers[component] = X[labels == component].mean(axis=0)
                else:
                    centers[component] = X[rng.randint(n_samples)]
        return centers, labels

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        n_samples, n_features = X.shape
        nk = responsibilities.sum(axis=0) + 10.0 * np.finfo(float).eps
        weights = nk / n_samples
        means = (responsibilities.T @ X) / nk[:, None]
        covariances = np.empty((self.n_components, n_features, n_features), dtype=float)

        for component in range(self.n_components):
            centered = X - means[component]
            weighted_centered = responsibilities[:, component][:, None] * centered
            covariances[component] = (weighted_centered.T @ centered) / nk[component]
            covariances[component].flat[:: n_features + 1] += self.reg_covar

        return weights, means, covariances

    def _estimate_log_prob_resp(self, X: np.ndarray):
        return self._estimate_log_prob_resp_with_params(X, self.weights_, self.means_, self.covariances_)

    def _estimate_log_prob_resp_with_params(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ):
        weighted_log_prob = self._estimate_weighted_log_prob_with_params(X, weights, means, covariances)
        log_prob_norm = self._logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, None]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X: np.ndarray):
        return self._estimate_weighted_log_prob_with_params(X, self.weights_, self.means_, self.covariances_)

    def _estimate_weighted_log_prob_with_params(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> np.ndarray:
        log_prob = np.empty((len(X), self.n_components), dtype=float)
        for component in range(self.n_components):
            log_prob[:, component] = self._log_multivariate_normal_density(
                X,
                means[component],
                covariances[component],
            )
        return log_prob + np.log(weights + np.finfo(float).eps)

    def _log_multivariate_normal_density(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        n_features = X.shape[1]
        covariance = covariance.copy()
        for _ in range(5):
            try:
                chol = np.linalg.cholesky(covariance)
                break
            except np.linalg.LinAlgError:
                covariance.flat[:: n_features + 1] += self.reg_covar * 10.0
        else:
            chol = np.linalg.cholesky(covariance + 1e-3 * np.eye(n_features))

        centered = (X - mean).T
        solved = np.linalg.solve(chol, centered)
        log_det = 2.0 * np.sum(np.log(np.diagonal(chol)))
        quadratic = np.sum(solved**2, axis=0)
        return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + quadratic)

    def _n_parameters(self) -> int:
        mean_params = self.n_components * self.n_features_in_
        covariance_params = self.n_components * self.n_features_in_ * (self.n_features_in_ + 1) // 2
        weight_params = self.n_components - 1
        return int(mean_params + covariance_params + weight_params)

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
        maximum = np.max(values, axis=axis, keepdims=True)
        stable = values - maximum
        summed = np.sum(np.exp(stable), axis=axis, keepdims=True)
        return np.squeeze(maximum + np.log(summed), axis=axis)
