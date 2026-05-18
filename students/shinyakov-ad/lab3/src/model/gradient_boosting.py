from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: list[DecisionTreeRegressor] = []
        self.alphas: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).astype(float)

        F = np.zeros(X.shape[0])
        self.trees = []
        self.alphas = []

        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            g = y - p

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            tree.fit(X, g)

            pred = tree.predict(X)

            def objective(alpha):
                p_alpha = self._sigmoid(F + alpha * pred)
                return -np.mean(
                    y * np.log(p_alpha) + (1 - y) * np.log(1 - p_alpha)
                )

            result = minimize_scalar(
                objective,
                bounds=(0, 10),
                method="bounded",
            )

            alpha = result.x * self.learning_rate

            F += alpha * pred

            self.trees.append(tree)
            self.alphas.append(alpha)

        return self

    def predict_proba(self, X: np.ndarray):
        X = np.asarray(X)

        F = np.zeros(X.shape[0], dtype=float)

        for tree, alpha in zip(self.trees, self.alphas):
            F += alpha * tree.predict(X)

        return self._sigmoid(F)

    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))