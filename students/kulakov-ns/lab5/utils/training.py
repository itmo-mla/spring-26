from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterGrid

from utils.dataset import build_user_item_matrix
from utils.metrics import rmse_on_interactions


SLIM_PARAM_GRID = {
    "alpha_l1": [0.001, 0.005, 0.01],
    "alpha_l2": [0.001, 0.01],
}

LSA_PARAM_GRID = {
    "n_components": [4, 6, 8, 10],
}


@dataclass
class SimpleGridSearchResult:
    best_estimator_: Any
    best_params_: Dict[str, Any]
    best_score_: float
    cv_results_: List[Dict[str, Any]] = field(default_factory=list)


def get_cv(random_state: int = 42) -> KFold:
    return KFold(n_splits=5, shuffle=True, random_state=random_state)


def _fit_and_score(estimator, train_part, valid_part, n_users, n_items, user_to_index, item_to_index):
    matrix = build_user_item_matrix(train_part, n_users, n_items, user_to_index, item_to_index)
    estimator.fit(matrix)
    prediction = np.clip(estimator.predict_matrix(matrix), 1.0, 5.0)
    return rmse_on_interactions(prediction, valid_part)


def fit_grid_search(
    estimator,
    train_interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    user_to_index: Dict[str, int],
    item_to_index: Dict[str, int],
    param_grid: Dict[str, List[Any]],
) -> SimpleGridSearchResult:
    train_interactions = train_interactions.reset_index(drop=True)
    cv = get_cv()

    best_params = None
    best_score = np.inf
    cv_results = []

    for params in ParameterGrid(param_grid):
        scores = []
        for train_index, valid_index in cv.split(train_interactions):
            train_part = train_interactions.iloc[train_index].copy()
            valid_part = train_interactions.iloc[valid_index].copy()

            model = clone(estimator)
            model.set_params(**params)
            score = _fit_and_score(model, train_part, valid_part, n_users, n_items, user_to_index, item_to_index)
            scores.append(float(score))

        mean_score = float(np.mean(scores))
        cv_results.append({
            "params": dict(params),
            "mean_valid_rmse": mean_score,
            "split_scores": scores,
        })

        if mean_score < best_score:
            best_score = mean_score
            best_params = dict(params)

    full_matrix = build_user_item_matrix(train_interactions, n_users, n_items, user_to_index, item_to_index)
    best_estimator = clone(estimator)
    best_estimator.set_params(**best_params)
    best_estimator.fit(full_matrix)

    return SimpleGridSearchResult(
        best_estimator_=best_estimator,
        best_params_=best_params,
        best_score_=best_score,
        cv_results_=cv_results,
    )
