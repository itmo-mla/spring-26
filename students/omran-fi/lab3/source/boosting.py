from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """Binary gradient boosting classifier with logistic loss.

    The boosting loop is implemented manually. DecisionTreeRegressor is used as
    the weak learner so the experiment focuses on additive boosting mechanics.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
        eps = 1e-15
        proba = np.clip(proba, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(proba) + (1.0 - y_true) * np.log(1.0 - proba)))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Only binary targets encoded as 0/1 are supported.")

        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.train_loss_ = []

        rng = np.random.default_rng(self.random_state)
        p = np.clip(np.mean(y), 1e-15, 1.0 - 1e-15)
        self.init_prediction_ = float(np.log(p / (1.0 - p)))
        raw_predictions = np.full(X.shape[0], self.init_prediction_, dtype=float)

        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(raw_predictions)
            residuals = y - probabilities

            if self.subsample < 1.0:
                sample_size = max(1, int(X.shape[0] * self.subsample))
                sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)
            else:
                sample_idx = np.arange(X.shape[0])

            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
            )
            tree.fit(X[sample_idx], residuals[sample_idx])
            raw_predictions += self.learning_rate * tree.predict(X)

            self.estimators_.append(tree)
            self.train_loss_.append(self._log_loss(y, self._sigmoid(raw_predictions)))

        self.feature_importances_ = self._feature_importances()
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X)
        raw = np.full(X.shape[0], self.init_prediction_, dtype=float)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict_proba(self, X) -> np.ndarray:
        proba_1 = self._sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - proba_1, proba_1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _feature_importances(self) -> np.ndarray:
        values = np.zeros(self.n_features_in_, dtype=float)
        if not self.estimators_:
            return values
        for tree in self.estimators_:
            values += tree.feature_importances_
        values /= len(self.estimators_)
        total = values.sum()
        if total > 0:
            values /= total
        return values
