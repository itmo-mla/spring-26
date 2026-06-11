from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor


class BinaryGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """Gradient boosting for binary classification with logistic loss."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "BinaryGradientBoostingClassifier":
        X_frame = pd.DataFrame(X).reset_index(drop=True)
        X_array = X_frame.to_numpy(dtype=float)
        y_array = np.asarray(y)

        self.classes_ = np.sort(np.unique(y_array))
        if len(self.classes_) != 2:
            raise ValueError("BinaryGradientBoostingClassifier supports exactly two classes.")
        y_binary = (y_array == self.classes_[1]).astype(float)

        self.n_features_in_ = X_array.shape[1]
        self.feature_names_in_ = np.asarray(X_frame.columns, dtype=object)
        self.estimators_: list[DecisionTreeRegressor] = []
        self.leaf_values_: list[dict[int, float]] = []
        self.train_loss_: list[float] = []

        positive_rate = np.clip(y_binary.mean(), 1e-6, 1 - 1e-6)
        self.init_score_ = float(np.log(positive_rate / (1 - positive_rate)))
        raw_predictions = np.full(len(y_binary), self.init_score_, dtype=float)
        rng = np.random.RandomState(self.random_state)

        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in (0, 1].")

        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(raw_predictions)
            residuals = y_binary - probabilities

            if self.subsample < 1.0:
                sample_size = max(1, int(round(len(y_binary) * self.subsample)))
                sample_indices = rng.choice(len(y_binary), size=sample_size, replace=False)
            else:
                sample_indices = np.arange(len(y_binary))

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.randint(0, np.iinfo(np.int32).max)),
            )
            tree.fit(X_array[sample_indices], residuals[sample_indices])

            train_leaves = tree.apply(X_array)
            leaf_values = self._compute_leaf_values(train_leaves, y_binary, probabilities)
            raw_predictions += self.learning_rate * self._map_leaf_values(train_leaves, leaf_values)

            self.estimators_.append(tree)
            self.leaf_values_.append(leaf_values)
            self.train_loss_.append(self._log_loss(y_binary, raw_predictions))

        self.feature_importances_ = self._mean_feature_importances()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_fitted()
        raw_predictions = self._raw_predict(X)
        positive_proba = self._sigmoid(raw_predictions)
        return np.column_stack((1 - positive_proba, positive_proba))

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return self.classes_[(probabilities >= 0.5).astype(int)]

    def score(self, X: Any, y: Any) -> float:
        return float(accuracy_score(y, self.predict(X)))

    def staged_predict_proba(self, X: Any) -> list[np.ndarray]:
        self._check_fitted()
        X_array = pd.DataFrame(X).to_numpy(dtype=float)
        raw_predictions = np.full(len(X_array), self.init_score_, dtype=float)
        staged_probabilities = []

        for tree, leaf_values in zip(self.estimators_, self.leaf_values_):
            leaves = tree.apply(X_array)
            raw_predictions += self.learning_rate * self._map_leaf_values(leaves, leaf_values)
            positive_proba = self._sigmoid(raw_predictions)
            staged_probabilities.append(np.column_stack((1 - positive_proba, positive_proba)))

        return staged_probabilities

    def _raw_predict(self, X: Any) -> np.ndarray:
        X_array = pd.DataFrame(X).to_numpy(dtype=float)
        raw_predictions = np.full(len(X_array), self.init_score_, dtype=float)
        for tree, leaf_values in zip(self.estimators_, self.leaf_values_):
            leaves = tree.apply(X_array)
            raw_predictions += self.learning_rate * self._map_leaf_values(leaves, leaf_values)
        return raw_predictions

    def _compute_leaf_values(
        self,
        leaves: np.ndarray,
        y_binary: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict[int, float]:
        values = {}
        for leaf_id in np.unique(leaves):
            mask = leaves == leaf_id
            numerator = float(np.sum(y_binary[mask] - probabilities[mask]))
            denominator = float(np.sum(probabilities[mask] * (1 - probabilities[mask])))
            values[int(leaf_id)] = numerator / denominator if denominator > 1e-12 else 0.0
        return values

    def _map_leaf_values(self, leaves: np.ndarray, leaf_values: dict[int, float]) -> np.ndarray:
        return np.asarray([leaf_values.get(int(leaf_id), 0.0) for leaf_id in leaves], dtype=float)

    def _mean_feature_importances(self) -> np.ndarray:
        importances = np.zeros(self.n_features_in_, dtype=float)
        for tree in self.estimators_:
            importances += tree.feature_importances_
        importances /= len(self.estimators_)
        total = importances.sum()
        return importances / total if total > 0 else importances

    def _check_fitted(self) -> None:
        if not hasattr(self, "estimators_") or not self.estimators_:
            raise RuntimeError("The model is not fitted yet.")

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))

    @staticmethod
    def _log_loss(y_binary: np.ndarray, raw_predictions: np.ndarray) -> float:
        probabilities = np.clip(BinaryGradientBoostingClassifier._sigmoid(raw_predictions), 1e-15, 1 - 1e-15)
        return float(-np.mean(y_binary * np.log(probabilities) + (1 - y_binary) * np.log(1 - probabilities)))
