from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import ParameterGrid


@dataclass
class GridSearchResult:
    params: dict[str, Any]
    score: float


class GridSearchEstimator:
    def __init__(
        self,
        estimator_class: type,
        param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
        fixed_params: dict[str, Any] | None = None,
    ) -> None:
        self.estimator_class = estimator_class
        self.param_grid = param_grid
        self.fixed_params = fixed_params or {}

        self.best_estimator_: Any | None = None
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.results_: list[GridSearchResult] = []


    def _build_estimator(self, params: dict[str, Any]) -> Any:
        estimator_params = {**self.fixed_params, **params}
        return self.estimator_class(**estimator_params)


    def _score(self, estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
        return estimator.compute_oob_score(X, y)


    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearchEstimator":
        best_score = -np.inf

        for params in ParameterGrid(self.param_grid):
            estimator = self._build_estimator(params)
            estimator.fit(X, y)
            score = self._score(estimator, X, y)

            self.results_.append(GridSearchResult(params=params, score=score))

            if score > best_score:
                best_score = score
                self.best_estimator_ = estimator
                self.best_params_ = params
                self.best_score_ = score

        return self
