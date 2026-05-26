from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


class MyGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """Gradient boosting regressor с MSE loss и regression tree в качестве базовой модели."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> MyGradientBoostingRegressor:
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        self.initial_pred_ = float(np.mean(y))
        F = np.full(n, self.initial_pred_)
        self.estimators_: list[DecisionTreeRegressor] = []
        self.train_loss_: list[float] = []

        for _ in range(self.n_estimators):
            # negative gradient of MSE: r_i = y_i - F(x_i)
            residuals = y - F

            if self.subsample < 1.0:
                size = max(1, int(n * self.subsample))
                idx = rng.choice(n, size=size, replace=False)
                X_sub, r_sub = X[idx], residuals[idx]
            else:
                X_sub, r_sub = X, residuals

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            tree.fit(X_sub, r_sub)
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean((y - F) ** 2)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous targets."""
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.initial_pred_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.estimators_:
            return np.array([])
        total = sum(t.feature_importances_ for t in self.estimators_)
        s = float(total.sum())
        return total / s if s > 0.0 else total


class MyGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosting classifier через log-loss (только бинарная классификация)."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> MyGradientBoostingClassifier:
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"Only binary classification supported, got {len(self.classes_)} classes."
            )
        y_enc = (y == self.classes_[1]).astype(float)
        n = len(y_enc)

        # initial log-odds
        p0 = float(np.clip(np.mean(y_enc), 1e-7, 1.0 - 1e-7))
        self.initial_pred_ = float(np.log(p0 / (1.0 - p0)))
        F = np.full(n, self.initial_pred_)
        self.estimators_: list[DecisionTreeRegressor] = []
        self.train_loss_: list[float] = []

        for _ in range(self.n_estimators):
            # negative gradient of log-loss: r_i = y_i - sigmoid(F(x_i))
            p = _sigmoid(F)
            residuals = y_enc - p

            if self.subsample < 1.0:
                size = max(1, int(n * self.subsample))
                idx = rng.choice(n, size=size, replace=False)
                X_sub, r_sub = X[idx], residuals[idx]
            else:
                X_sub, r_sub = X, residuals

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            tree.fit(X_sub, r_sub)
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)

            p_new = _sigmoid(F)
            log_loss = -float(
                np.mean(
                    y_enc * np.log(np.clip(p_new, 1e-7, 1.0))
                    + (1.0 - y_enc) * np.log(np.clip(1.0 - p_new, 1e-7, 1.0))
                )
            )
            self.train_loss_.append(log_loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.initial_pred_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        p1 = _sigmoid(F)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.estimators_:
            return np.array([])
        total = sum(t.feature_importances_ for t in self.estimators_)
        s = float(total.sum())
        return total / s if s > 0.0 else total
