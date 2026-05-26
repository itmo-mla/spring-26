from dataclasses import dataclass

import numpy as np


@dataclass
class _TreeNode:
    value: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class RegressionTree:
    """Small CART-style regression tree for squared-error splits"""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_thresholds: int = 64,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.root_: _TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegressionTree":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain the same number of rows")

        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root_ = self._build_node(x, y, depth=0)
        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("tree is not fitted")

        x = np.asarray(x, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in x])

    def _build_node(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        value = float(np.mean(y))
        node = _TreeNode(value=value)

        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or y.size < 2 * self.min_samples_leaf
            or np.allclose(y, y[0])
        ):
            return node

        split = self._find_best_split(x, y)
        if split is None:
            return node

        feature_index, threshold, gain = split
        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_node(x[left_mask], y[left_mask], depth + 1)
        node.right = self._build_node(x[right_mask], y[right_mask], depth + 1)
        if self.feature_importances_ is not None:
            self.feature_importances_[feature_index] += gain
        return node

    def _find_best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float, float] | None:
        best_feature: int | None = None
        best_threshold: float | None = None
        best_gain = 0.0
        parent_error = self._squared_error(y)
        n_samples = y.size

        for feature_index in range(x.shape[1]):
            values = x[:, feature_index]
            order = np.argsort(values)
            sorted_values = values[order]
            sorted_y = y[order]

            split_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:]) + 1
            if split_positions.size == 0:
                continue
            if split_positions.size > self.max_thresholds:
                indices = np.linspace(0, split_positions.size - 1, self.max_thresholds, dtype=int)
                split_positions = split_positions[np.unique(indices)]

            valid_positions = split_positions[
                (split_positions >= self.min_samples_leaf) & (n_samples - split_positions >= self.min_samples_leaf)
            ]
            if valid_positions.size == 0:
                continue

            prefix_sum = np.cumsum(sorted_y)
            prefix_square_sum = np.cumsum(sorted_y**2)
            total_sum = prefix_sum[-1]
            total_square_sum = prefix_square_sum[-1]

            for split_position in valid_positions:
                left_count = split_position
                right_count = n_samples - split_position

                left_sum = prefix_sum[split_position - 1]
                left_square_sum = prefix_square_sum[split_position - 1]
                right_sum = total_sum - left_sum
                right_square_sum = total_square_sum - left_square_sum

                left_error = left_square_sum - left_sum**2 / left_count
                right_error = right_square_sum - right_sum**2 / right_count
                gain = parent_error - left_error - right_error

                if gain > best_gain:
                    best_feature = feature_index
                    best_threshold = float((sorted_values[split_position - 1] + sorted_values[split_position]) / 2.0)
                    best_gain = float(gain)

        if best_feature is None or best_threshold is None:
            return None
        return best_feature, best_threshold, best_gain

    @staticmethod
    def _squared_error(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        centered = y - np.mean(y)
        return float(np.dot(centered, centered))

    def _predict_row(self, row: np.ndarray, node: _TreeNode) -> float:
        if node.is_leaf:
            return node.value

        if node.feature_index is None or node.threshold is None:
            return node.value
        next_node = node.left if row[node.feature_index] <= node.threshold else node.right
        if next_node is None:
            return node.value
        return self._predict_row(row, next_node)


class GradientBoostingBinaryClassifier:
    """Gradient boosting for binary classification with logistic loss"""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.15,
        max_depth: int = 3,
        min_samples_split: int = 8,
        min_samples_leaf: int = 5,
        max_thresholds: int = 48,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.initial_prediction_: float | None = None
        self.trees_: list[RegressionTree] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostingBinaryClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("y must contain binary labels encoded as 0 and 1")

        positive_rate = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.initial_prediction_ = float(np.log(positive_rate / (1 - positive_rate)))
        raw_predictions = np.full(y.shape, self.initial_prediction_, dtype=float)
        self.trees_ = []

        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(raw_predictions)
            residuals = y - probabilities
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_thresholds=self.max_thresholds,
            )
            tree.fit(x, residuals)
            raw_predictions += self.learning_rate * tree.predict(x)
            self.trees_.append(tree)

        self.feature_importances_ = self._collect_feature_importances(x.shape[1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = self._sigmoid(self._raw_predict(x))
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        if self.initial_prediction_ is None:
            raise ValueError("model is not fitted")

        x = np.asarray(x, dtype=float)
        raw_predictions = np.full(x.shape[0], self.initial_prediction_, dtype=float)
        for tree in self.trees_:
            raw_predictions += self.learning_rate * tree.predict(x)
        return raw_predictions

    def _collect_feature_importances(self, n_features: int) -> np.ndarray:
        importances = np.zeros(n_features, dtype=float)
        for tree in self.trees_:
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        total_importance = importances.sum()
        if total_importance > 0:
            importances /= total_importance
        return importances

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -50, 50)
        return 1.0 / (1.0 + np.exp(-clipped))
