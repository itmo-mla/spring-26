from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNet


class SlimRecommender:
    """Sparse Linear Method with non-negative elastic-net coordinate descent."""

    def __init__(
        self,
        l1: float = 0.002,
        l2: float = 0.02,
        max_iter: int = 80,
        tol: float = 1e-5,
        positive: bool = True,
    ) -> None:
        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive

    def fit(self, matrix: np.ndarray) -> "SlimRecommender":
        matrix = np.asarray(matrix, dtype=float)
        self.train_matrix_ = matrix
        n_items = matrix.shape[1]
        self.coef_ = np.zeros((n_items, n_items), dtype=float)
        column_norms = np.sum(matrix * matrix, axis=0) + self.l2

        for target_item in range(n_items):
            y = matrix[:, target_item]
            if np.count_nonzero(y) == 0:
                continue
            self.coef_[:, target_item] = self._fit_target(matrix, y, target_item, column_norms)
        return self

    def predict_all(self, matrix: np.ndarray | None = None) -> np.ndarray:
        if matrix is None:
            matrix = self.train_matrix_
        return np.asarray(matrix, dtype=float) @ self.coef_

    def _fit_target(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_item: int,
        column_norms: np.ndarray,
    ) -> np.ndarray:
        n_items = X.shape[1]
        weights = np.zeros(n_items, dtype=float)
        residual = y.copy()

        for _ in range(self.max_iter):
            max_change = 0.0
            for item in range(n_items):
                if item == target_item or column_norms[item] <= 0:
                    continue
                column = X[:, item]
                old_weight = weights[item]
                if old_weight != 0.0:
                    residual += column * old_weight

                rho = float(column @ residual)
                if self.positive:
                    new_weight = max(0.0, rho - self.l1) / column_norms[item]
                else:
                    new_weight = np.sign(rho) * max(0.0, abs(rho) - self.l1) / column_norms[item]

                if new_weight != 0.0:
                    residual -= column * new_weight
                weights[item] = new_weight
                max_change = max(max_change, abs(new_weight - old_weight))

            if max_change < self.tol:
                break

        weights[target_item] = 0.0
        return weights


class SklearnSlimRecommender:
    """Reference SLIM-style model using sklearn ElasticNet for each item."""

    def __init__(self, alpha: float = 0.001, l1_ratio: float = 0.35, max_iter: int = 3000) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def fit(self, matrix: np.ndarray) -> "SklearnSlimRecommender":
        matrix = np.asarray(matrix, dtype=float)
        self.train_matrix_ = matrix
        n_items = matrix.shape[1]
        self.coef_ = np.zeros((n_items, n_items), dtype=float)

        for target_item in range(n_items):
            y = matrix[:, target_item]
            if np.count_nonzero(y) == 0:
                continue
            X = np.delete(matrix, target_item, axis=1)
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=False,
                positive=True,
                max_iter=self.max_iter,
                selection="cyclic",
                tol=1e-5,
            )
            model.fit(X, y)
            coef = np.insert(model.coef_, target_item, 0.0)
            self.coef_[:, target_item] = coef
        return self

    def predict_all(self, matrix: np.ndarray | None = None) -> np.ndarray:
        if matrix is None:
            matrix = self.train_matrix_
        return np.asarray(matrix, dtype=float) @ self.coef_


class NumpyLSARecommender:
    """Latent semantic model based on explicit dense truncated SVD."""

    def __init__(self, n_components: int = 20) -> None:
        self.n_components = n_components

    def fit(self, matrix: np.ndarray) -> "NumpyLSARecommender":
        matrix = np.asarray(matrix, dtype=float)
        self.train_matrix_ = matrix
        U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
        k = min(self.n_components, len(singular_values))
        self.user_factors_ = U[:, :k] * singular_values[:k]
        self.item_factors_ = Vt[:k, :]
        self.singular_values_ = singular_values[:k]
        return self

    def predict_all(self, matrix: np.ndarray | None = None) -> np.ndarray:
        if matrix is None:
            user_factors = self.user_factors_
        else:
            matrix = np.asarray(matrix, dtype=float)
            user_factors = matrix @ self.item_factors_.T
        prediction = user_factors @ self.item_factors_
        return np.clip(prediction, 0.0, None)


class SklearnLSARecommender:
    """Reference latent semantic model using sklearn TruncatedSVD."""

    def __init__(self, n_components: int = 20, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, matrix: np.ndarray) -> "SklearnLSARecommender":
        matrix = np.asarray(matrix, dtype=float)
        self.train_matrix_ = matrix
        self.model_ = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.user_factors_ = self.model_.fit_transform(matrix)
        self.singular_values_ = self.model_.singular_values_
        return self

    def predict_all(self, matrix: np.ndarray | None = None) -> np.ndarray:
        if matrix is None or matrix is self.train_matrix_:
            user_factors = self.user_factors_
        else:
            user_factors = self.model_.transform(matrix)
        prediction = self.model_.inverse_transform(user_factors)
        return np.clip(prediction, 0.0, None)
