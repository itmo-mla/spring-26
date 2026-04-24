import numpy as np
from typing import List, Tuple, Optional
from node import TreeNode


class CustomDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5, min_gain=0.01):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.n_classes = 0
        self.feature_names = []

    def _gini_impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _calculate_gain(self, y: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray) -> float:
        n = len(y)
        if n == 0:
            return 0.0

        parent_impurity = self._gini_impurity(y)

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            return 0.0

        left_impurity = self._gini_impurity(y[left_mask])
        right_impurity = self._gini_impurity(y[right_mask])

        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        return parent_impurity - weighted_impurity

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            valid_mask = ~np.isnan(values)

            if np.sum(valid_mask) < self.min_samples_split:
                continue

            unique_vals = np.unique(values[valid_mask])
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            if len(thresholds) == 0:
                thresholds = unique_vals

            for threshold in thresholds:
                left_mask = valid_mask & (values >= threshold)
                right_mask = valid_mask & (values < threshold)

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gain = self._calculate_gain(y[valid_mask], left_mask[valid_mask], right_mask[valid_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode()
        node.samples_count = len(y)

        # Распределение классов для листа
        classes, counts = np.unique(y, return_counts=True)
        node.class_distribution = np.zeros(self.n_classes)
        for c, cnt in zip(classes, counts):
            node.class_distribution[int(c)] = cnt
        node.class_label = classes[np.argmax(counts)]

        # Условия остановки
        if (depth >= self.max_depth or
                len(np.unique(y)) == 1 or
                len(y) < self.min_samples_split):
            node.is_leaf = True
            return node

        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        if best_feature is None or best_gain < self.min_gain:
            node.is_leaf = True
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.is_leaf = False

        # Разделение данных
        values = X[:, best_feature]
        valid_mask = ~np.isnan(values)
        left_mask = valid_mask & (values >= best_threshold)
        right_mask = valid_mask & (values < best_threshold)

        total_valid = np.sum(valid_mask)
        if total_valid > 0:
            node.q_left = np.sum(left_mask) / total_valid
            node.q_right = np.sum(right_mask) / total_valid

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        self.n_classes = len(np.unique(y))
        self.feature_names = feature_names
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_proba_single(self, x: np.ndarray, node: TreeNode) -> np.ndarray:
        if node.is_leaf:
            total = np.sum(node.class_distribution)
            if total == 0:
                return np.zeros(self.n_classes)
            return node.class_distribution / total

        val = x[node.feature_index]

        if np.isnan(val):
            prob_left = self._predict_proba_single(x, node.left)
            prob_right = self._predict_proba_single(x, node.right)

            return node.q_left * prob_left + node.q_right * prob_right
        else:
            if val >= node.threshold:
                return self._predict_proba_single(x, node.left)
            else:
                return self._predict_proba_single(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_proba_single(x, self.root) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
