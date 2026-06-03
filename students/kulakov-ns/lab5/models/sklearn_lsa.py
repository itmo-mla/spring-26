from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import TruncatedSVD

from utils.training import LSA_PARAM_GRID, fit_grid_search


class SklearnLatentSemanticRecommender(BaseEstimator, RegressorMixin):
    """Эталонная латентная модель на базе TruncatedSVD из scikit-learn."""

    def __init__(self, n_components: int = 6, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        n_users, n_items = matrix.shape
        self.n_components_ = int(min(self.n_components, max(1, min(n_users, n_items) - 1)))

        user_mean = np.zeros(n_users, dtype=float)
        centered = matrix.copy()
        for user_index in range(n_users):
            observed = matrix[user_index] > 0
            if np.any(observed):
                user_mean[user_index] = float(matrix[user_index, observed].mean())
                centered[user_index, observed] -= user_mean[user_index]

        self.svd_ = TruncatedSVD(n_components=self.n_components_, random_state=self.random_state)
        self.user_factors_ = self.svd_.fit_transform(centered)
        self.user_mean_ = user_mean
        return self

    def predict_matrix(self, X=None):
        if X is None:
            return self.svd_.inverse_transform(self.user_factors_) + self.user_mean_[:, None]

        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        centered = matrix.copy()
        user_mean = np.zeros(matrix.shape[0], dtype=float)
        for user_index in range(matrix.shape[0]):
            observed = matrix[user_index] > 0
            if np.any(observed):
                user_mean[user_index] = float(matrix[user_index, observed].mean())
                centered[user_index, observed] -= user_mean[user_index]

        user_factors = self.svd_.transform(centered)
        return self.svd_.inverse_transform(user_factors) + user_mean[:, None]

    def score(self, X, y=None):
        matrix = pd.DataFrame(X).to_numpy(dtype=float)
        prediction = self.predict_matrix(matrix)
        observed = matrix > 0
        if not np.any(observed):
            return 0.0
        rmse = np.sqrt(np.mean((matrix[observed] - prediction[observed]) ** 2))
        return -float(rmse)


def get_sklearn_lsa(dataset):
    estimator = SklearnLatentSemanticRecommender(random_state=42)
    return fit_grid_search(
        estimator=estimator,
        train_interactions=dataset["train_interactions"],
        n_users=dataset["n_users"],
        n_items=dataset["n_items"],
        user_to_index=dataset["user_to_index"],
        item_to_index=dataset["item_to_index"],
        param_grid=LSA_PARAM_GRID,
    )
