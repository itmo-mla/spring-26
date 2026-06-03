from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegressionNode:
    value: float
    depth: int
    n_samples: int
    feature_index: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: "RegressionNode | None" = None
    right: "RegressionNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class CARTRegressor:
    """Small numeric CART regressor used as a weak learner in boosting."""

    def __init__(
        self,
        max_depth: int = 2,
        min_samples_split: int = 8,
        min_samples_leaf: int = 4,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y) -> "CARTRegressor":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        self.n_features_in_ = self.X_.shape[1]
        self.rng_ = np.random.RandomState(self.random_state)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        self.root_ = self._build(np.arange(len(self.y_)), depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in X], dtype=float)

    def _build(self, indexes: np.ndarray, depth: int) -> RegressionNode:
        y = self.y_[indexes]
        node = RegressionNode(value=float(np.mean(y)), depth=depth, n_samples=len(indexes))

        if (
            depth >= self.max_depth
            or len(indexes) < self.min_samples_split
            or len(indexes) < 2 * self.min_samples_leaf
            or np.var(y) <= 1e-12
        ):
            return node

        split = self._best_split(indexes)
        if split is None:
            return node

        feature_index, threshold, gain, left_indexes, right_indexes = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.gain = gain
        self.feature_importances_[feature_index] += gain
        node.left = self._build(left_indexes, depth + 1)
        node.right = self._build(right_indexes, depth + 1)
        return node

    def _best_split(self, indexes: np.ndarray):
        parent_sse = self._sse(self.y_[indexes])
        best = None

        for feature_index in self._candidate_features():
            values = self.X_[indexes, feature_index]
            order = np.argsort(values, kind="mergesort")
            sorted_indexes = indexes[order]
            sorted_values = values[order]
            sorted_y = self.y_[sorted_indexes]

            if sorted_values[0] == sorted_values[-1]:
                continue

            prefix_sum = np.cumsum(sorted_y)
            prefix_sq_sum = np.cumsum(sorted_y**2)
            total_sum = prefix_sum[-1]
            total_sq_sum = prefix_sq_sum[-1]
            n = len(sorted_y)

            for split_pos in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                if sorted_values[split_pos - 1] == sorted_values[split_pos]:
                    continue

                left_n = split_pos
                right_n = n - split_pos
                left_sum = prefix_sum[split_pos - 1]
                left_sq_sum = prefix_sq_sum[split_pos - 1]
                right_sum = total_sum - left_sum
                right_sq_sum = total_sq_sum - left_sq_sum

                left_sse = left_sq_sum - (left_sum * left_sum) / left_n
                right_sse = right_sq_sum - (right_sum * right_sum) / right_n
                gain = parent_sse - left_sse - right_sse

                if gain <= 1e-12:
                    continue

                if best is None or gain > best[2]:
                    threshold = float((sorted_values[split_pos - 1] + sorted_values[split_pos]) / 2.0)
                    best = (
                        int(feature_index),
                        threshold,
                        float(gain),
                        sorted_indexes[:split_pos],
                        sorted_indexes[split_pos:],
                    )

        return best

    def _candidate_features(self) -> np.ndarray:
        n_features = self.n_features_in_
        if self.max_features is None:
            count = n_features
        elif self.max_features == "sqrt":
            count = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            count = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            count = max(1, int(np.ceil(self.max_features * n_features)))
        else:
            count = int(self.max_features)

        count = min(n_features, max(1, count))
        if count == n_features:
            return np.arange(n_features)
        return self.rng_.choice(n_features, size=count, replace=False)

    @staticmethod
    def _sse(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.sum((y - np.mean(y)) ** 2))

    def _predict_row(self, row: np.ndarray, node: RegressionNode) -> float:
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
