"""
Кастомная модель гауссовой смеси, обученная с помощью EM-алгоритма.

Реализует интерфейс, совместимый с sklearn, соответствующий подмножеству
`sklearn.mixture.GaussianMixture`. Поддерживает четыре типа ковариации и
несколько стратегий инициализации. Все вычисления вероятностей выполняются
в логарифмическом пространстве с использованием `scipy.special.logsumexp` для обеспечения численной устойчивости.

EM-алгоритм: 
- на E-шаге вычисляются апостериорные веса
- на M-шаге обновляются веса, средние и ковариации по аналитической формуле, 
  а сходимость отслеживается по среднему логарифмическому правдоподобию по выборке.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

from .covariance import (
    estimate_covariances_diag,
    estimate_covariances_full,
    estimate_covariances_spherical,
    estimate_covariances_tied,
    log_prob_diag,
    log_prob_full,
    log_prob_spherical,
    log_prob_tied,
    n_parameters,
)

CovarianceType = Literal["full", "diag", "tied", "spherical"]
InitType = Literal["random", "kmeans", "kmeans++"]


@dataclass
class _SingleRunResult:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    lower_bound: float
    n_iter: int
    converged: bool
    loss_history: list[float]


class MyGaussianMixture(BaseEstimator):

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovarianceType = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: InitType = "kmeans",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose

    #============================#
    #=========== fit ============#
    #============================#
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> MyGaussianMixture:
        X = self._validate_X(X, reset=True)
        rng = np.random.default_rng(self.random_state)
        best: _SingleRunResult | None = None
        
        for _ in range(self.n_init):
            seed = int(rng.integers(0, 2**31 - 1))
            result = self._fit_single(X, seed=seed)
            if best is None or result.lower_bound > best.lower_bound:
                best = result
        assert best is not None

        self.weights_ = best.weights
        self.means_ = best.means
        self.covariances_ = best.covariances
        self.n_iter_ = best.n_iter
        self.lower_bound_ = best.lower_bound
        self.converged_ = best.converged
        self.loss_history_ = best.loss_history
        
        return self


    def _fit_single(self, X: np.ndarray, seed: int) -> _SingleRunResult:
        weights, means, covariances = self._initialize_parameters(X, seed)
        prev_ll = -np.inf
        loss_history: list[float] = []
        converged = False

        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            log_prob, log_resp = self._e_step(X, weights, means, covariances)
            ll = float(log_prob.mean())
            loss_history.append(ll)

            if n_iter > 1 and ll < prev_ll - self.tol:
                if self.verbose:
                    print(f"[gmm] log-likelihood decreased at iter {n_iter}: {prev_ll:.6f} -> {ll:.6f}")

            if n_iter > 1 and abs(ll - prev_ll) < self.tol:
                converged = True
                prev_ll = ll
                break
            
            weights, means, covariances = self._m_step(X, np.exp(log_resp))
            prev_ll = ll

        return _SingleRunResult(
            weights=weights,
            means=means,
            covariances=covariances,
            lower_bound=prev_ll,
            n_iter=n_iter,
            converged=converged,
            loss_history=loss_history,
        )

    #============================#
    #====== initialization ======#
    #============================#
    def _initialize_parameters(self, X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = X.shape
        K = self.n_components
        rng = np.random.default_rng(seed)
        if self.init_params == "random":
            indices = rng.choice(n_samples, size=K, replace=False)
            means = X[indices]
            weights = np.full(K, 1.0 / K)
            resp = np.zeros((n_samples, K))
            # soft assign to nearest center
            d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(d2, axis=1)
            resp[np.arange(n_samples), nearest] = 1.0
        elif self.init_params in ("kmeans", "kmeans++"):
            kmeans_init = "k-means++" if self.init_params == "kmeans++" else "random"
            km = KMeans(n_clusters=K, n_init=1, init=kmeans_init, random_state=seed, max_iter=50)
            labels = km.fit_predict(X)
            means = km.cluster_centers_
            resp = np.zeros((n_samples, K))
            resp[np.arange(n_samples), labels] = 1.0
            weights = resp.sum(axis=0) / n_samples
        else:
            raise ValueError(f"unknown init_params: {self.init_params}")
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        covariances = self._compute_covariances(X, resp, means, nk)
        weights = nk / n_samples
        return weights, means, covariances

    def _compute_covariances(
        self,
        X: np.ndarray,
        resp: np.ndarray,
        means: np.ndarray,
        nk: np.ndarray,
    ) -> np.ndarray:
        if self.covariance_type == "full":
            return estimate_covariances_full(X, resp, means, nk, self.reg_covar)
        if self.covariance_type == "diag":
            return estimate_covariances_diag(X, resp, means, nk, self.reg_covar)
        if self.covariance_type == "tied":
            return estimate_covariances_tied(X, resp, means, nk, self.reg_covar)
        if self.covariance_type == "spherical":
            return estimate_covariances_spherical(X, resp, means, nk, self.reg_covar)
        raise ValueError(f"unknown covariance_type: {self.covariance_type}")

    #============================#
    #=========== E/M ============#
    #============================#
    def _e_step(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        log_prob_components = self._log_prob_components(X, means, covariances)
        weighted_log_prob = log_prob_components + np.log(weights)[None, :]
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)  # (n_samples,)
        log_resp = weighted_log_prob - log_prob_norm[:, None]
        return log_prob_norm, log_resp

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = (resp.T @ X) / nk[:, None]
        covariances = self._compute_covariances(X, resp, means, nk)
        weights = nk / n_samples
        return weights, means, covariances

    def _log_prob_components(self, X: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        if self.covariance_type == "full":
            return log_prob_full(X, means, covariances, self.reg_covar)
        if self.covariance_type == "diag":
            return log_prob_diag(X, means, covariances)
        if self.covariance_type == "tied":
            return log_prob_tied(X, means, covariances, self.reg_covar)
        if self.covariance_type == "spherical":
            return log_prob_spherical(X, means, covariances)
        raise ValueError(f"unknown covariance_type: {self.covariance_type}")

    #============================#
    #========= predict ==========#
    #============================#
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        log_prob_components = self._log_prob_components(X, self.means_, self.covariances_)
        weighted = log_prob_components + np.log(self.weights_)[None, :]
        return logsumexp(weighted, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        return float(self.score_samples(X).mean())

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        log_prob_components = self._log_prob_components(X, self.means_, self.covariances_)
        weighted = log_prob_components + np.log(self.weights_)[None, :]
        log_norm = logsumexp(weighted, axis=1, keepdims=True)
        return np.exp(weighted - log_norm)

    def sample(self, n_samples: int = 1, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        rng = np.random.default_rng(random_state if random_state is not None else self.random_state)
        components = rng.choice(self.n_components, size=n_samples, p=self.weights_)
        D = self.means_.shape[1]
        samples = np.empty((n_samples, D))
        for k in range(self.n_components):
            mask = components == k
            count = int(mask.sum())
            if count == 0:
                continue
            mean_k = self.means_[k]
            cov_k = self._component_full_cov(k)
            samples[mask] = rng.multivariate_normal(mean_k, cov_k, size=count)
        return samples, components

    def _component_full_cov(self, k: int) -> np.ndarray:
        if self.covariance_type == "full":
            return self.covariances_[k]
        D = self.means_.shape[1]
        if self.covariance_type == "diag":
            return np.diag(self.covariances_[k])
        if self.covariance_type == "tied":
            return self.covariances_
        if self.covariance_type == "spherical":
            return self.covariances_[k] * np.eye(D)
        raise ValueError(self.covariance_type)

    #============================#
    #======== AIC / BIC =========#
    #============================#
    def _n_parameters(self) -> int:
        return n_parameters(self.covariance_type, self.n_components, self.means_.shape[1])

    def bic(self, X: np.ndarray) -> float:
        log_lik = self.score(X) * X.shape[0]
        return -2.0 * log_lik + self._n_parameters() * np.log(X.shape[0])

    def aic(self, X: np.ndarray) -> float:
        log_lik = self.score(X) * X.shape[0]
        return -2.0 * log_lik + 2.0 * self._n_parameters()

    # === utility === #
    def _validate_X(self, X: np.ndarray, reset: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if reset:
            self.n_features_in_ = X.shape[1]
        else:
            if X.shape[1] != getattr(self, "n_features_in_", X.shape[1]):
                raise ValueError(
                    f"feature mismatch: model expects {self.n_features_in_}, got {X.shape[1]}"
                )
        return X


class MyGMMClassifier(BaseEstimator):
    """
    Генеративный классификатор с одним GMM на класс.

    Предсказывает argmax по `log P(y=c) + log p(x | y=c)`. 
    При K=1 и `covariance_type="diag"` математически это эквивалентно Gaussian Naive Bayes; 
    при K=1 и `covariance_type="full"` это эквивалентно QDA.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovarianceType = "diag",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: InitType = "kmeans",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> MyGMMClassifier:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self.models_: dict[object, MyGaussianMixture] = {}
        rng = np.random.default_rng(self.random_state)

        for cls in self.classes_:
            seed = int(rng.integers(0, 2**31 - 1))
            gmm = MyGaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                tol=self.tol,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=self.n_init,
                init_params=self.init_params,
                random_state=seed,
            )
            gmm.fit(X[y == cls])
            self.models_[cls] = gmm

        self.n_features_in_ = X.shape[1]
        return self


    def _joint_log_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        log_priors = np.log(self.class_prior_)
        out = np.empty((X.shape[0], len(self.classes_)))
        for j, cls in enumerate(self.classes_):
            out[:, j] = log_priors[j] + self.models_[cls].score_samples(X)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_proba(X)
        return self.classes_[np.argmax(joint, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_proba(X)
        log_norm = logsumexp(joint, axis=1, keepdims=True)
        return np.exp(joint - log_norm)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
