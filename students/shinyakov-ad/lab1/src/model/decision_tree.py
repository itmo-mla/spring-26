from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class DecisionNode:
    feature_index: int | None = None
    children: dict[Any, "DecisionNode"] | None = None
    branch_probs: dict[Any, float] | None = None
    value: Any | None = None
    positive_prob: float | None = None

class DecisionTree():

    def __init__(self, criterion = "gini"):
        self.tree = None
        self.criterion = criterion
    
    def fit(self, X, y):
        available_features = tuple(range(X.shape[1]))
        self.tree = self._build_tree(X, y, available_features=available_features)
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

    def _prune_node(self, node, X_val, y_val):
        if node is None or node.value is not None or not node.children:
            return

        for child in node.children.values():
            self._prune_node(child, X_val, y_val)

        acc_before = self._accuracy(X_val, y_val)

        old_state = (node.value, node.children, node.branch_probs, node.feature_index)

        node.value = int(node.positive_prob >= 0.5)
        node.children = None
        node.branch_probs = None
        node.feature_index = None

        if self._accuracy(X_val, y_val) < acc_before:
            node.value, node.children, node.branch_probs, node.feature_index = old_state

    def _predict_proba_one(self, x, node=None):
        if node is None:
            node = self.tree

        if node.value is not None:
            return float(node.value)

        feature_value = x[node.feature_index]

        if pd.isna(feature_value):
            if not node.children or not node.branch_probs:
                return node.positive_prob

            proba = 0.0
            total_weight = 0.0

            for cat, child in node.children.items():
                weight = node.branch_probs.get(cat, 0.0)
                proba += weight * self._predict_proba_one(x, child)
                total_weight += weight

            if total_weight == 0:
                return node.positive_prob

            return proba / total_weight

        if feature_value not in node.children:
            return node.positive_prob

        return self._predict_proba_one(x, node.children[feature_value])
    
    def _build_tree(self, X, y, depth=0, available_features=None):
        if X.shape[0] == 0:
            return None

        if available_features is None:
            available_features = tuple(range(X.shape[1]))

        if len(available_features) == 0:
            values, counts = np.unique(y, return_counts=True)
            majority = values[np.argmax(counts)]
            return DecisionNode(
                value=majority,
                positive_prob=float(np.mean(y))
            )

        if len(np.unique(y)) == 1:
            return DecisionNode(
                value=y[0],
                positive_prob=float(y[0])
            )

        gains = {}

        for feature in available_features:
            column = X[:, feature]

            valid_mask = ~pd.isna(column)
            valid_column = column[valid_mask]
            valid_y = y[valid_mask]

            if len(valid_y) == 0:
                gains[feature] = -np.inf
                continue

            categories = np.unique(valid_column)
            gain = 0.0

            for cat in categories:
                mask = valid_column == cat
                y_subset = valid_y[mask]
                gain += self._calculate_gain(valid_y, y_subset)

            gains[feature] = self._count_entropy(np.mean(valid_y)) - gain

        best_feature = max(gains, key=gains.get)

        column = X[:, best_feature]
        valid_mask = ~pd.isna(column)
        valid_column = column[valid_mask]
        if len(valid_column) == 0:
            values, counts = np.unique(y, return_counts=True)
            majority = values[np.argmax(counts)]
            return DecisionNode(
                value=majority,
                positive_prob=float(np.mean(y))
            )

        categories, counts = np.unique(valid_column, return_counts=True)
        branch_probs = {
            cat: count / counts.sum()
            for cat, count in zip(categories, counts)
        }

        node = DecisionNode(
            feature_index=best_feature,
            children={},
            branch_probs=branch_probs,
            positive_prob=float(np.mean(y))
        )
        next_features = tuple(f for f in available_features if f != best_feature)

        for cat in categories:
            mask = column == cat
            X_subset = X[mask]
            y_subset = y[mask]

            child = self._build_tree(
                X_subset,
                y_subset,
                depth + 1,
                available_features=next_features,
            )
            node.children[cat] = child

        return node
    
    def _calculate_gain(self, y, y_subset):

        weight = len(y_subset) / len(y)
        children_entropy = weight * self._count_entropy(np.mean(y_subset))

        return children_entropy
    
    def _count_entropy(self, p):
        if self.criterion == "gini":
            return 4 * p * (1 - p)
        
        raise "No criterion found"
