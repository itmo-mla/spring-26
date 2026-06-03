from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator

from utils.training import fit_grid_search


class GaussianMixtureCustom(BaseEstimator):
    def __init__(
        self,
        n_components: int = 3,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        n_init: int = 5,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def _initialize_parameters(self, X: np.ndarray, rng: np.random.RandomState):
        n_samples, n_features = X.shape
        selected = rng.choice(n_samples, self.n_components, replace=False)
        means = X[selected].copy()

        empirical_covariance = np.cov(X.T, bias=True)
        if empirical_covariance.ndim == 0:
            empirical_covariance = np.array([[float(empirical_covariance)]])
        empirical_covariance += self.reg_covar * np.eye(n_features)
        covariances = np.repeat(empirical_covariance[None, :, :], self.n_components, axis=0)

        weights = np.full(self.n_components, 1.0 / self.n_components)
        return weights, means, covariances

    @staticmethod
    def _estimate_log_gaussian_prob(X: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        n_components = means.shape[0]
        log_prob = np.empty((n_samples, n_components), dtype=float)

        for component_index in range(n_components):
            centered = X - means[component_index]
            covariance = covariances[component_index]
            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise np.linalg.LinAlgError("Ковариационная матрица должна быть положительно определённой")
            precision = np.linalg.inv(covariance)
            mahalanobis = np.einsum("ij,jk,ik->i", centered, precision, centered)
            log_prob[:, component_index] = -0.5 * (
                n_features * np.log(2.0 * np.pi) + log_det + mahalanobis
            )

        return log_prob

    def _e_step(self, X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray):
        weighted_log_prob = self._estimate_log_gaussian_prob(X, means, covariances) + np.log(weights)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_responsibilities = weighted_log_prob - log_prob_norm[:, None]
        responsibilities = np.exp(log_responsibilities)
        lower_bound = float(log_prob_norm.mean())
        return responsibilities, lower_bound

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        n_samples, n_features = X.shape
        nk = responsibilities.sum(axis=0) + 10.0 * np.finfo(float).eps
        weights = nk / n_samples
        means = (responsibilities.T @ X) / nk[:, None]

        covariances = np.empty((self.n_components, n_features, n_features), dtype=float)
        for component_index in range(self.n_components):
            centered = X - means[component_index]
            weighted_centered = centered * responsibilities[:, component_index][:, None]
            covariance = (weighted_centered.T @ centered) / nk[component_index]
            covariance += self.reg_covar * np.eye(n_features)
            covariances[component_index] = covariance

        return weights, means, covariances

    def _fit_once(self, X: np.ndarray, seed: int):
        rng = np.random.RandomState(seed)
        weights, means, covariances = self._initialize_parameters(X, rng)

        lower_bound = -np.inf
        converged = False
        n_iter = 0

        for iteration in range(1, self.max_iter + 1):
            responsibilities, new_lower_bound = self._e_step(X, weights, means, covariances)
            weights, means, covariances = self._m_step(X, responsibilities)

            if abs(new_lower_bound - lower_bound) <= self.tol:
                converged = True
                lower_bound = new_lower_bound
                n_iter = iteration
                break

            lower_bound = new_lower_bound
            n_iter = iteration

        return {
            "weights": weights,
            "means": means,
            "covariances": covariances,
            "lower_bound": lower_bound,
            "converged": converged,
            "n_iter": n_iter,
        }

    def fit(self, X, y=None):
        X = pd.DataFrame(X).to_numpy(dtype=float)
        n_samples, n_features = X.shape
        if self.n_components <= 0 or self.n_components > n_samples:
            raise ValueError("Число компонент должно быть положительным и не больше числа объектов")

        self.n_features_in_ = n_features
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 10**9, size=self.n_init)

        best_run = None
        for seed in seeds:
            run = self._fit_once(X, int(seed))
            if best_run is None or run["lower_bound"] > best_run["lower_bound"]:
                best_run = run

        self.weights_ = best_run["weights"]
        self.means_ = best_run["means"]
        self.covariances_ = best_run["covariances"]
        self.lower_bound_ = float(best_run["lower_bound"])
        self.converged_ = bool(best_run["converged"])
        self.n_iter_ = int(best_run["n_iter"])
        return self

    def _estimate_weighted_log_prob(self, X: np.ndarray) -> np.ndarray:
        return self._estimate_log_gaussian_prob(X, self.means_, self.covariances_) + np.log(self.weights_)

    def score_samples(self, X):
        X = pd.DataFrame(X).to_numpy(dtype=float)
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        return logsumexp(weighted_log_prob, axis=1)

    def score(self, X, y=None):
        return float(self.score_samples(X).mean())

    def predict_proba(self, X):
        X = pd.DataFrame(X).to_numpy(dtype=float)
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        return np.exp(weighted_log_prob - log_prob_norm[:, None])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _n_parameters(self) -> int:
        n_features = self.n_features_in_
        covariance_params = self.n_components * n_features * (n_features + 1) // 2
        mean_params = self.n_components * n_features
        weight_params = self.n_components - 1
        return covariance_params + mean_params + weight_params

    def bic(self, X):
        X = pd.DataFrame(X).to_numpy(dtype=float)
        log_likelihood = float(self.score_samples(X).sum())
        return -2.0 * log_likelihood + self._n_parameters() * np.log(X.shape[0])

    def aic(self, X):
        X = pd.DataFrame(X).to_numpy(dtype=float)
        log_likelihood = float(self.score_samples(X).sum())
        return -2.0 * log_likelihood + 2.0 * self._n_parameters()



def get_my_gmm(X_train: pd.DataFrame):
    estimator = GaussianMixtureCustom(random_state=42)
    return fit_grid_search(estimator, X_train)
