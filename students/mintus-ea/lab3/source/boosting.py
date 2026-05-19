from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from tree import CARTRegressor


class LogisticGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """Binary gradient boosting for logistic loss with custom CART weak learners."""

    def __init__(
        self,
        n_estimators: int = 140,
        learning_rate: float = 0.08,
        max_depth: int = 2,
        min_samples_split: int = 8,
        min_samples_leaf: int = 4,
        subsample: float = 1.0,
        max_features: int | float | str | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y) -> "LogisticGradientBoostingClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("LogisticGradientBoostingClassifier supports binary labels 0/1 only.")

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.estimators_: list[CARTRegressor] = []
        self.train_loss_: list[float] = []
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        rng = np.random.RandomState(self.random_state)

        positive_rate = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
        self.initial_raw_score_ = float(np.log(positive_rate / (1.0 - positive_rate)))
        raw_score = np.full(len(y), self.initial_raw_score_, dtype=float)

        for _ in range(self.n_estimators):
            probability = self._sigmoid(raw_score)
            residual = y - probability

            if self.subsample < 1.0:
                sample_size = max(2, int(len(y) * self.subsample))
                indexes = rng.choice(len(y), size=sample_size, replace=False)
            else:
                indexes = np.arange(len(y))

            tree_seed = int(rng.randint(0, np.iinfo(np.int32).max))
            tree = CARTRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.fit(X[indexes], residual[indexes])
            update = tree.predict(X)
            raw_score += self.learning_rate * update

            self.estimators_.append(tree)
            self.feature_importances_ += tree.feature_importances_
            self.train_loss_.append(self._log_loss(y, raw_score))

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        raw_score = np.full(len(X), self.initial_raw_score_, dtype=float)
        for tree in self.estimators_:
            raw_score += self.learning_rate * tree.predict(X)
        return raw_score

    def predict_proba(self, X) -> np.ndarray:
        positive = self._sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y) -> float:
        return accuracy_score(y, self.predict(X))

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-values))

    def _log_loss(self, y: np.ndarray, raw_score: np.ndarray) -> float:
        probability = np.clip(self._sigmoid(raw_score), 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability)))
