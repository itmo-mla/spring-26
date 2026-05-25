from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet

from utils.training import SLIM_PARAM_GRID, fit_grid_search


class SklearnSlimRecommender(BaseEstimator, RegressorMixin):
    """Эталонная SLIM-модель, где item-item регрессии решаются ElasticNet из scikit-learn."""

    def __init__(
        self,
        alpha_l1: float = 0.005,
        alpha_l2: float = 0.01,
        max_iter: int = 2000,
        tol: float = 1e-5,
        positive: bool = True,
        random_state: int = 42,
    ):
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.random_state = random_state

    @staticmethod
    def _center_by_user_mean(matrix: np.ndarray):
        user_mean = np.zeros(matrix.shape[0], dtype=float)
        centered = np.zeros_like(matrix, dtype=float)
        for user_index in range(matrix.shape[0]):
            observed = matrix[user_index] > 0
            if np.any(observed):
                user_mean[user_index] = float(matrix[user_index, observed].mean())
                centered[user_index, observed] = matrix[user_index, observed] - user_mean[user_index]
        return centered, user_mean

    def fit(self, X, y=None):
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        centered, user_mean = self._center_by_user_mean(matrix)
        n_items = matrix.shape[1]
        coefficients = np.zeros((n_items, n_items), dtype=float)

        alpha = self.alpha_l1 + self.alpha_l2
        l1_ratio = self.alpha_l1 / alpha if alpha > 0 else 0.0

        for item_index in range(n_items):
            rated_users = matrix[:, item_index] > 0
            if not np.any(rated_users):
                continue

            target = centered[rated_users, item_index]
            predictors = centered[rated_users].copy()
            predictors[:, item_index] = 0.0

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=False,
                positive=self.positive,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                selection="cyclic",
            )
            model.fit(predictors, target)
            coefficients[:, item_index] = model.coef_

        np.fill_diagonal(coefficients, 0.0)
        self.coef_ = coefficients
        self.user_mean_ = user_mean
        return self

    def predict_matrix(self, X=None):
        if X is None:
            raise ValueError("Для SLIM нужно передать user-item матрицу")
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        centered, user_mean = self._center_by_user_mean(matrix)
        return centered @ self.coef_ + user_mean[:, None]

    def score(self, X, y=None):
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        prediction = self.predict_matrix(matrix)
        observed = matrix > 0
        if not np.any(observed):
            return 0.0
        rmse = np.sqrt(np.mean((matrix[observed] - prediction[observed]) ** 2))
        return -float(rmse)


def get_sklearn_slim(dataset):
    estimator = SklearnSlimRecommender(random_state=42)
    return fit_grid_search(
        estimator=estimator,
        train_interactions=dataset["train_interactions"],
        n_users=dataset["n_users"],
        n_items=dataset["n_items"],
        user_to_index=dataset["user_to_index"],
        item_to_index=dataset["item_to_index"],
        param_grid=SLIM_PARAM_GRID,
    )
