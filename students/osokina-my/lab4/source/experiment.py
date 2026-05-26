from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .data import load_dataset
from .gmm import GaussianMixtureModel
from .metrics import aic, bic, count_gmm_params, log_likelihood


@dataclass
class ExperimentResult:
    n_components: int
    custom_score: float
    sklearn_score: float
    custom_log_likelihood: float
    sklearn_log_likelihood: float
    custom_aic: float
    sklearn_aic: float
    custom_bic: float
    sklearn_bic: float
    custom_n_iter: int
    sklearn_n_iter: int


def run_experiments(
    component_range: range | list[int] | None = None,
    random_state: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if component_range is None:
        component_range = range(2, 6)

    X, y, feature_names, target_names = load_dataset()
    results: list[ExperimentResult] = []

    n_samples, n_features = X.shape
    for k in component_range:
        custom = GaussianMixtureModel(
            n_components=k,
            random_state=random_state,
            max_iter=max_iter,
        ).fit(X)

        ref = GaussianMixture(
            n_components=k,
            random_state=random_state,
            max_iter=max_iter,
            reg_covar=1e-6,
            init_params="random",
        ).fit(X)

        ll_custom = log_likelihood(custom.score_samples(X))
        ll_sklearn = float(ref.score(X) * n_samples)
        n_params = count_gmm_params(n_features, k)

        results.append(
            ExperimentResult(
                n_components=k,
                custom_score=custom.score(X),
                sklearn_score=float(ref.score(X)),
                custom_log_likelihood=ll_custom,
                sklearn_log_likelihood=ll_sklearn,
                custom_aic=aic(ll_custom, n_params),
                sklearn_aic=float(aic(ll_sklearn, n_params)),
                custom_bic=bic(ll_custom, n_params, n_samples),
                sklearn_bic=float(bic(ll_sklearn, n_params, n_samples)),
                custom_n_iter=custom.n_iter_,
                sklearn_n_iter=int(ref.n_iter_),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in results])
    return X, y, df
