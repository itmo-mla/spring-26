from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class DecisionNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "DecisionNode | None" = None
    right: "DecisionNode | None" = None
    branch_probs: tuple[float, float] | None = None
    value: Any | None = None
    positive_prob: float | None = None

class DecisionTree():

    def __init__(self, criterion = "gini"):
        self.tree = None
        self.criterion = criterion
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        w = np.ones(len(y), dtype=float)
        available_features = tuple(range(X.shape[1]))
        self.tree = self._build_tree(X, y, w, available_features=available_features)
        return self

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def predict_proba(self, X):
        probs = np.asarray([self._predict_proba_one(x) for x in X], dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])

    def prune_inner(self, X_val, y_val):
        if self.tree is None:
            return self

        X_val = np.asarray(X_val, dtype=object)
        y_val = np.asarray(y_val)

        self._prune_node(self.tree, X_val, y_val)
        return self
    
    def _predict_one(self, x):
        proba = self._predict_proba_one(x)
        return int(proba >= 0.5)

    def _accuracy(self, X, y):
        y_pred = np.asarray(self.predict(X))
        return float(np.mean(y_pred == y))

    def _f1(self, X, y):
        y_true = np.asarray(y)
        y_pred = np.asarray(self.predict(X))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))

    def _prune_node(self, node, X_val, y_val):
        if node is None or node.value is not None:
            return

        self._prune_node(node.left, X_val, y_val)
        self._prune_node(node.right, X_val, y_val)

        acc_before = self._accuracy(X_val, y_val)
        f1_before = self._f1(X_val, y_val)

        old_state = (
            node.value,
            node.left,
            node.right,
            node.branch_probs,
            node.feature_index,
            node.threshold,
        )

        node.value = int(node.positive_prob >= 0.5)
        node.left = None
        node.right = None
        node.branch_probs = None
        node.feature_index = None
        node.threshold = None

        acc_after = self._accuracy(X_val, y_val)
        f1_after = self._f1(X_val, y_val)
        if acc_after < acc_before or f1_after < f1_before:
            (
                node.value,
                node.left,
                node.right,
                node.branch_probs,
                node.feature_index,
                node.threshold,
            ) = old_state

    def _predict_proba_one(self, x, node=None):
        if node is None:
            node = self.tree

        if node.value is not None:
            return float(node.positive_prob if node.positive_prob is not None else node.value)

        feature_value = x[node.feature_index]

        if pd.isna(feature_value):
            if node.left is None or node.right is None or node.branch_probs is None:
                return node.positive_prob

            left_prob = self._predict_proba_one(x, node.left)
            right_prob = self._predict_proba_one(x, node.right)
            return node.branch_probs[0] * left_prob + node.branch_probs[1] * right_prob

        numeric_value = pd.to_numeric(pd.Series([feature_value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            return node.positive_prob

        if numeric_value <= node.threshold:
            if node.left is None:
                return node.positive_prob
            return self._predict_proba_one(x, node.left)

        if node.right is None:
            return node.positive_prob
        return self._predict_proba_one(x, node.right)
    
    def _build_tree(self, X, y, w, depth=0, available_features=None):
        if X.shape[0] == 0 or np.sum(w) <= 0:
            return None

        if available_features is None:
            available_features = tuple(range(X.shape[1]))

        if len(available_features) == 0:
            majority = self._majority_class(y, w)
            return DecisionNode(
                value=majority,
                positive_prob=self._weighted_mean(y, w)
            )

        if len(np.unique(y)) == 1:
            return DecisionNode(
                value=y[0],
                positive_prob=float(y[0])
            )

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in available_features:
            column = pd.to_numeric(pd.Series(X[:, feature]), errors="coerce").to_numpy()
            valid_mask = ~np.isnan(column)
            valid_column = column[valid_mask]
            valid_y = y[valid_mask]
            valid_w = w[valid_mask]

            if len(valid_y) == 0 or np.sum(valid_w) <= 0:
                continue

            unique_vals = np.unique(valid_column)
            if len(unique_vals) < 2:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            parent_impurity = self._count_entropy(self._weighted_mean(valid_y, valid_w))
            for threshold in thresholds:
                left_mask = valid_column <= threshold
                right_mask = valid_column > threshold
                left_y = valid_y[left_mask]
                right_y = valid_y[right_mask]
                left_w = valid_w[left_mask]
                right_w = valid_w[right_mask]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gain = parent_impurity - self._calculate_gain(valid_y, valid_w, left_y, left_w) - self._calculate_gain(
                    valid_y, valid_w, right_y, right_w
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)

        if best_feature is None:
            majority = self._majority_class(y, w)
            return DecisionNode(
                value=majority,
                positive_prob=self._weighted_mean(y, w)
            )

        column = pd.to_numeric(pd.Series(X[:, best_feature]), errors="coerce").to_numpy()
        valid_mask = ~np.isnan(column)
        missing_mask = ~valid_mask
        left_mask = valid_mask & (column <= best_threshold)
        right_mask = valid_mask & (column > best_threshold)
        left_weight = np.sum(w[left_mask])
        right_weight = np.sum(w[right_mask])
        total_count = left_weight + right_weight
        if total_count == 0:
            majority = self._majority_class(y, w)
            return DecisionNode(
                value=majority,
                positive_prob=self._weighted_mean(y, w)
            )
        branch_probs = (left_weight / total_count, right_weight / total_count)

        node = DecisionNode(
            feature_index=best_feature,
            threshold=best_threshold,
            branch_probs=branch_probs,
            positive_prob=self._weighted_mean(y, w)
        )
        next_features = tuple(f for f in available_features if f != best_feature)

        X_left_obs = X[left_mask]
        y_left_obs = y[left_mask]
        w_left_obs = w[left_mask]
        X_right_obs = X[right_mask]
        y_right_obs = y[right_mask]
        w_right_obs = w[right_mask]

        if np.any(missing_mask):
            X_missing = X[missing_mask]
            y_missing = y[missing_mask]
            w_missing = w[missing_mask]

            X_left = np.concatenate([X_left_obs, X_missing], axis=0)
            y_left = np.concatenate([y_left_obs, y_missing], axis=0)
            w_left = np.concatenate([w_left_obs, w_missing * branch_probs[0]], axis=0)

            X_right = np.concatenate([X_right_obs, X_missing], axis=0)
            y_right = np.concatenate([y_right_obs, y_missing], axis=0)
            w_right = np.concatenate([w_right_obs, w_missing * branch_probs[1]], axis=0)
        else:
            X_left, y_left, w_left = X_left_obs, y_left_obs, w_left_obs
            X_right, y_right, w_right = X_right_obs, y_right_obs, w_right_obs

        node.left = self._build_tree(
            X_left,
            y_left,
            w_left,
            depth + 1,
            available_features=next_features,
        )
        node.right = self._build_tree(
            X_right,
            y_right,
            w_right,
            depth + 1,
            available_features=next_features,
        )

        if node.left is None:
            majority = self._majority_class(y, w)
            node.left = DecisionNode(value=majority, positive_prob=self._weighted_mean(y, w))
        if node.right is None:
            majority = self._majority_class(y, w)
            node.right = DecisionNode(value=majority, positive_prob=self._weighted_mean(y, w))

        return node
    
    def _calculate_gain(self, y, w, y_subset, w_subset):
        total = np.sum(w)
        if total <= 0:
            return 0.0
        subset_total = np.sum(w_subset)
        if subset_total <= 0:
            return 0.0
        weight = subset_total / total
        children_entropy = weight * self._count_entropy(self._weighted_mean(y_subset, w_subset))

        return children_entropy

    def _weighted_mean(self, y, w):
        total = np.sum(w)
        if total <= 0:
            return 0.5
        return float(np.sum(y * w) / total)

    def _majority_class(self, y, w):
        weight_pos = np.sum(w[y == 1])
        weight_neg = np.sum(w[y == 0])
        return 1 if weight_pos >= weight_neg else 0
    
    def _count_entropy(self, p):
        if self.criterion == "gini":
            return 4 * p * (1 - p)
        
        raise "No criterion found"
