from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from utils.training import LSA_PARAM_GRID, fit_grid_search


class LatentSemanticRecommenderCustom(BaseEstimator, RegressorMixin):
    """Латентная семантическая модель на основе усечённого SVD user-item матрицы."""

    def __init__(self, n_components: int = 6, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        n_users, n_items = matrix.shape
        max_components = max(1, min(n_users, n_items))
        self.n_components_ = int(min(self.n_components, max_components))

        user_mean = np.zeros(n_users, dtype=float)
        centered = matrix.copy()
        for user_index in range(n_users):
            observed = matrix[user_index] > 0
            if np.any(observed):
                user_mean[user_index] = float(matrix[user_index, observed].mean())
                centered[user_index, observed] -= user_mean[user_index]

        u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.n_components_]
        self.singular_values_ = singular_values[: self.n_components_]
        self.user_factors_ = u[:, : self.n_components_] * self.singular_values_
        self.user_mean_ = user_mean
        return self

    def predict_matrix(self, X=None):
        if X is None:
            prediction = self.user_factors_ @ self.components_
            return prediction + self.user_mean_[:, None]

        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        centered = matrix.copy()
        user_mean = np.zeros(matrix.shape[0], dtype=float)
        for user_index in range(matrix.shape[0]):
            observed = matrix[user_index] > 0
            if np.any(observed):
                user_mean[user_index] = float(matrix[user_index, observed].mean())
                centered[user_index, observed] -= user_mean[user_index]

        user_factors = centered @ self.components_.T
        return user_factors @ self.components_ + user_mean[:, None]

    def score(self, X, y=None):
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        prediction = self.predict_matrix(matrix)
        observed = matrix > 0
        if not np.any(observed):
            return 0.0
        rmse = np.sqrt(np.mean((matrix[observed] - prediction[observed]) ** 2))
        return -float(rmse)


def get_my_lsa(dataset):
    estimator = LatentSemanticRecommenderCustom(random_state=42)
    return fit_grid_search(
        estimator=estimator,
        train_interactions=dataset["train_interactions"],
        n_users=dataset["n_users"],
        n_items=dataset["n_items"],
        user_to_index=dataset["user_to_index"],
        item_to_index=dataset["item_to_index"],
        param_grid=LSA_PARAM_GRID,
    )
