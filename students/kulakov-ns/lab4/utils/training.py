from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterGrid
from threadpoolctl import threadpool_limits


PARAM_GRID = {
    "n_components": [2, 3, 4],
    "reg_covar": [1e-6, 1e-4, 1e-3],
}


@dataclass
class SimpleGridSearchResult:
    best_estimator_: Any
    best_params_: Dict[str, Any]
    best_score_: float
    cv_results_: List[Dict[str, Any]] = field(default_factory=list)



def get_cv(random_state: int = 42) -> KFold:
    return KFold(n_splits=5, shuffle=True, random_state=random_state)



def fit_grid_search(estimator, X_train) -> SimpleGridSearchResult:
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    cv = get_cv()

    best_estimator = None
    best_params = None
    best_score = -np.inf
    cv_results = []

    for params in ParameterGrid(PARAM_GRID):
        scores = []
        for train_index, valid_index in cv.split(X_train):
            X_fold_train = X_train.iloc[train_index]
            X_fold_valid = X_train.iloc[valid_index]

            model = clone(estimator)
            model.set_params(**params)
            with threadpool_limits(limits=1):
                model.fit(X_fold_train)
                scores.append(float(model.score(X_fold_valid)))

        mean_score = float(np.mean(scores))
        cv_results.append({
            "params": dict(params),
            "mean_test_score": mean_score,
            "split_scores": scores,
        })

        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    best_estimator = clone(estimator)
    best_estimator.set_params(**best_params)
    with threadpool_limits(limits=1):
        best_estimator.fit(X_train)

    return SimpleGridSearchResult(
        best_estimator_=best_estimator,
        best_params_=best_params,
        best_score_=best_score,
        cv_results_=cv_results,
    )
