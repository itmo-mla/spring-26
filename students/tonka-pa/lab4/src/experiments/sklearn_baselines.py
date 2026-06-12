"""
Базовые модели sklearn для оценки плотности и классификации.

Для классификации мы обертываем sklearn.mixture.GaussianMixture для каждого класса в небольшой хелпер, 
чтобы он имел тот же интерфейс fit/predict/predict_proba, что и `MyGMMClassifier`. 
`GaussianNB` и `QuadraticDiscriminantAnalysis` используются в качестве дополнительных 
эталонных классификаторов, поскольку они соответствуют вырожденным случаям GMM-классификатора с K=1.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture as SkGaussianMixture
from sklearn.naive_bayes import GaussianNB


def make_density_baseline(
    n_components: int,
    covariance_type: str,
    random_state: int,
    n_init: int = 1,
    init_params: str = "kmeans",
    max_iter: int = 100,
    tol: float = 1e-3,
    reg_covar: float = 1e-6,
) -> SkGaussianMixture:
    return SkGaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
        init_params=init_params,
        max_iter=max_iter,
        tol=tol,
        reg_covar=reg_covar,
    )


class SklearnGMMClassifier(BaseEstimator, ClassifierMixin):
    """Per-class `sklearn.mixture.GaussianMixture` classifier."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "diag",
        n_init: int = 1,
        init_params: str = "kmeans",
        max_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.init_params = init_params
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnGMMClassifier:
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self.models_: dict[object, SkGaussianMixture] = {}
        rng = np.random.default_rng(self.random_state)
        for cls in self.classes_:
            gmm = SkGaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=int(rng.integers(0, 2**31 - 1)),
                n_init=self.n_init,
                init_params=self.init_params,
                max_iter=self.max_iter,
                tol=self.tol,
                reg_covar=self.reg_covar,
            )
            gmm.fit(X[y == cls])
            self.models_[cls] = gmm
        return self

    def _joint_log_proba(self, X: np.ndarray) -> np.ndarray:
        log_priors = np.log(self.class_prior_)
        out = np.empty((X.shape[0], len(self.classes_)))
        for j, cls in enumerate(self.classes_):
            out[:, j] = log_priors[j] + self.models_[cls].score_samples(X)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_proba(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_proba(X)
        log_norm = logsumexp(joint, axis=1, keepdims=True)
        return np.exp(joint - log_norm)


__all__ = [
    "GaussianNB",
    "QuadraticDiscriminantAnalysis",
    "SklearnGMMClassifier",
    "make_density_baseline",
]
