"""
Бинарное решающее дерево в духе жадного ID3: на каждом шаге выбирается сплит
с максимальным уменьшением неопределённости Джини
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .node import TreeNode
from .pruning import reduced_error_prune


class DecisionTreeID3:
    """Бинарное дерево: прирост по уменьшению неопределённости Джини (impurity gain)"""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: TreeNode | None = None
        self.feature_names = None
        self.n_classes_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return float(1.0 - np.sum(p**2))

    def _gain(self, y, left_mask, right_mask):
        n = len(y)
        n_left = int(np.sum(left_mask))
        n_right = int(np.sum(right_mask))
        if n_left == 0 or n_right == 0:
            return 0.0
        g_parent = self._gini(y)
        g_left = self._gini(y[left_mask])
        g_right = self._gini(y[right_mask])
        return g_parent - (n_left / n) * g_left - (n_right / n) * g_right

    def _best_split(self, X, y, feature_names):
        best_gain = -1.0
        best_split = None
        for idx, _name in enumerate(feature_names):
            col = X[:, idx]
            not_nan = pd.notna(col)
            if int(np.sum(not_nan)) < 2 * self.min_samples_split:
                continue
            X_sub = X[not_nan]
            y_sub = y[not_nan]
            col_sub = col[not_nan]

            if len(np.unique(col_sub)) < 2:
                continue

            is_cat = isinstance(col_sub[0], str) or (
                hasattr(col_sub, "dtype") and col_sub.dtype == object
            )
            if is_cat:
                categories = np.unique(col_sub)
                for cat in categories:
                    left_mask = col_sub == cat
                    right_mask = ~left_mask
                    if (
                        int(np.sum(left_mask)) < self.min_samples_split
                        or int(np.sum(right_mask)) < self.min_samples_split
                    ):
                        continue
                    gain = self._gain(y_sub, left_mask, right_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (idx, "cat", cat)
            else:
                uniq = np.sort(np.unique(col_sub.astype(float)))
                thresholds = (uniq[:-1] + uniq[1:]) / 2
                for thr in thresholds:
                    left_mask = col_sub <= thr
                    right_mask = ~left_mask
                    if (
                        int(np.sum(left_mask)) < self.min_samples_split
                        or int(np.sum(right_mask)) < self.min_samples_split
                    ):
                        continue
                    gain = self._gain(y_sub, left_mask, right_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (idx, "num", float(thr))
        return best_split, best_gain

    def _build_tree(self, X, y, depth, feature_names):
        if (
            len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            maj = int(np.bincount(y).argmax())
            return TreeNode(kind="leaf", leaf_class=maj, n_samples=len(y))

        split, _gain = self._best_split(X, y, feature_names)
        if split is None:
            maj = int(np.bincount(y).argmax())
            return TreeNode(kind="leaf", leaf_class=maj, n_samples=len(y))

        idx, split_type, value = split
        feature_name = feature_names[idx]
        col = X[:, idx]

        if split_type == "cat":
            left_mask_full = col == value
            right_mask_full = (col != value) & pd.notna(col)
        else:
            left_mask_full = (col <= value) & pd.notna(col)
            right_mask_full = (col > value) & pd.notna(col)

        n_total = len(y)
        n_left = int(np.sum(left_mask_full))
        n_right = int(np.sum(right_mask_full))

        left_child = self._build_tree(
            X[left_mask_full], y[left_mask_full], depth + 1, feature_names
        )
        right_child = self._build_tree(
            X[right_mask_full], y[right_mask_full], depth + 1, feature_names
        )
        maj_class = int(np.bincount(y).argmax())

        return TreeNode(
            kind="internal",
            feature=feature_name,
            split_type=split_type,
            value=value,
            left=left_child,
            right=right_child,
            weight_left=n_left / n_total if n_total > 0 else 0.0,
            weight_right=n_right / n_total if n_total > 0 else 0.0,
            majority_class=maj_class,
            n_samples=len(y),
        )

    def fit(self, X, y, feature_names):
        self.feature_names = list(feature_names)
        self.n_classes_ = int(np.max(y)) + 1
        self.tree = self._build_tree(np.asarray(X), np.asarray(y), 0, self.feature_names)

    @staticmethod
    def _branch_weights_normalized(weight_left: float, weight_right: float):
        """Доли объектов с известным значением признака в левой/правой ветви (сумма = 1)"""
        s = weight_left + weight_right
        if s <= 0:
            return 0.5, 0.5
        return weight_left / s, weight_right / s

    def _predict_one(self, x, node: TreeNode):
        if node.is_leaf:
            return node.leaf_class

        f_name = node.feature
        idx = self.feature_names.index(f_name)
        val = x[idx]
        n_cls = self.n_classes_

        if pd.isna(val):
            prob = np.zeros(n_cls, dtype=float)
            wl, wr = self._branch_weights_normalized(node.weight_left, node.weight_right)
            left_c = self._predict_one(x, node.left)
            right_c = self._predict_one(x, node.right)
            prob[left_c] += wl
            prob[right_c] += wr
            return int(np.argmax(prob))

        if node.split_type == "cat":
            if val == node.value:
                return self._predict_one(x, node.left)
            return self._predict_one(x, node.right)
        if val <= node.value:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        if self.feature_names is None or self.tree is None:
            raise ValueError("Модель не обучена")
        X = np.asarray(X)
        return np.array([self._predict_one(X[i], self.tree) for i in range(len(X))])

    def _predict_one_subtree(self, x, node: TreeNode):
        if node.is_leaf:
            return node.leaf_class
        f_name = node.feature
        idx = self.feature_names.index(f_name)
        val = x[idx]
        n_cls = self.n_classes_

        if pd.isna(val):
            prob = np.zeros(n_cls, dtype=float)
            wl, wr = self._branch_weights_normalized(node.weight_left, node.weight_right)
            left_c = self._predict_one_subtree(x, node.left)
            right_c = self._predict_one_subtree(x, node.right)
            prob[left_c] += wl
            prob[right_c] += wr
            return int(np.argmax(prob))

        if node.split_type == "cat":
            if val == node.value:
                return self._predict_one_subtree(x, node.left)
            return self._predict_one_subtree(x, node.right)
        if val <= node.value:
            return self._predict_one_subtree(x, node.left)
        return self._predict_one_subtree(x, node.right)

    def _predict_subtree(self, node: TreeNode, X_val):
        return np.array(
            [self._predict_one_subtree(X_val[i], node) for i in range(len(X_val))]
        )

    def prune(self, X_val, y_val):
        """Reduced Error Pruning с маской объектов валидации, дошедших до узла"""
        reduced_error_prune(self, X_val, y_val)
