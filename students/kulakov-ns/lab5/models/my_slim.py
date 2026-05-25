from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from utils.training import SLIM_PARAM_GRID, fit_grid_search


class SlimRecommenderCustom(BaseEstimator, RegressorMixin):
    """Учебная реализация SLIM через покоординатные item-item регрессии.

    Оптимизируется задача:
    0.5 / n * ||r_j - R w_j||^2 + alpha_l1 * ||w_j||_1 + 0.5 * alpha_l2 * ||w_j||^2,
    где диагональный коэффициент w_jj принудительно равен нулю.
    """

    def __init__(
        self,
        alpha_l1: float = 0.005,
        alpha_l2: float = 0.01,
        max_iter: int = 300,
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
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    def _fit_item(self, X: np.ndarray, y: np.ndarray, target_index: int) -> np.ndarray:
        n_samples, n_items = X.shape
        coefficients = np.zeros(n_items, dtype=float)

        X_work = X.copy()
        X_work[:, target_index] = 0.0

        spectral_norm = np.linalg.norm(X_work, ord=2)
        lipschitz = (spectral_norm ** 2) / max(n_samples, 1) + self.alpha_l2
        step = 1.0 / max(lipschitz, 1e-12)

        previous_objective = np.inf
        for _ in range(self.max_iter):
            residual = X_work @ coefficients - y
            gradient = (X_work.T @ residual) / max(n_samples, 1) + self.alpha_l2 * coefficients
            updated = coefficients - step * gradient
            updated = self._soft_threshold(updated, step * self.alpha_l1)
            updated[target_index] = 0.0
            if self.positive:
                updated = np.maximum(updated, 0.0)

            residual_updated = y - X_work @ updated
            objective = (
                0.5 * np.mean(residual_updated ** 2)
                + self.alpha_l1 * np.sum(np.abs(updated))
                + 0.5 * self.alpha_l2 * np.sum(updated ** 2)
            )

            if abs(previous_objective - objective) <= self.tol:
                coefficients = updated
                break

            coefficients = updated
            previous_objective = objective

        return coefficients

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
        self.train_shape_ = matrix.shape
        centered, user_mean = self._center_by_user_mean(matrix)
        n_items = matrix.shape[1]
        coefficients = np.zeros((n_items, n_items), dtype=float)

        for item_index in range(n_items):
            rated_users = matrix[:, item_index] > 0
            if not np.any(rated_users):
                continue
            target = centered[rated_users, item_index]
            predictors = centered[rated_users]
            coefficients[:, item_index] = self._fit_item(predictors, target, item_index)

        np.fill_diagonal(coefficients, 0.0)
        self.coef_ = coefficients
        self.user_mean_ = user_mean
        return self

    def predict_matrix(self, X=None):
        if X is None:
            raise ValueError("Для SLIM нужно передать user-item матрицу")
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        centered, user_mean = self._center_by_user_mean(matrix)
        prediction = centered @ self.coef_ + user_mean[:, None]
        return prediction

    def score(self, X, y=None):
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        prediction = self.predict_matrix(matrix)
        observed = matrix > 0
        if not np.any(observed):
            return 0.0
        rmse = np.sqrt(np.mean((matrix[observed] - prediction[observed]) ** 2))
        return -float(rmse)


def get_my_slim(dataset):
    estimator = SlimRecommenderCustom(random_state=42)
    return fit_grid_search(
        estimator=estimator,
        train_interactions=dataset["train_interactions"],
        n_users=dataset["n_users"],
        n_items=dataset["n_items"],
        user_to_index=dataset["user_to_index"],
        item_to_index=dataset["item_to_index"],
        param_grid=SLIM_PARAM_GRID,
    )
