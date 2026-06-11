from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    feature: str
    feature_type: str
    gain: float
    p_left: float
    p_right: float
    threshold: float | None = None
    category: Any | None = None


@dataclass
class Node:
    depth: int
    class_counts: np.ndarray
    prediction_index: int
    probabilities: np.ndarray
    weight: float
    is_leaf: bool = True
    feature: str | None = None
    feature_type: str | None = None
    threshold: float | None = None
    category: Any | None = None
    gain: float = 0.0
    p_left: float = 0.5
    p_right: float = 0.5
    left: "Node | None" = None
    right: "Node | None" = None


class ID3GiniClassifier:
    """Binary ID3 decision tree with Gini criterion and probabilistic missing routing."""

    def __init__(
        self,
        max_depth: int = 7,
        min_samples_split: int = 24,
        min_samples_leaf: int = 8,
        min_gain: float = 1e-5,
        max_thresholds: int = 64,
        prune_tolerance: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_thresholds = max_thresholds
        self.prune_tolerance = prune_tolerance

        self.root_: Node | None = None
        self.classes_: np.ndarray | None = None
        self.class_to_index_: dict[Any, int] = {}
        self.feature_types_: dict[str, str] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_types: dict[str, str],
    ) -> "ID3GiniClassifier":
        self.classes_ = np.sort(pd.Series(y).unique())
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        self.feature_types_ = dict(feature_types)

        X = X.reset_index(drop=True).copy()
        y_encoded = self._encode_y(y)
        weights = np.ones(len(X), dtype=float)
        self.root_ = self._build_node(X, y_encoded, weights, depth=0)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        X = X.reset_index(drop=True)
        return np.vstack([self._predict_row_proba(row, self.root_) for _, row in X.iterrows()])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        indexes = np.argmax(probabilities, axis=1)
        return self.classes_[indexes]

    def prune(self, X_val: pd.DataFrame, y_val: np.ndarray) -> int:
        """Reduced error pruning on validation data."""
        self._check_is_fitted()
        weights = np.ones(len(X_val), dtype=float)
        y_encoded = self._encode_y(y_val)
        return self._prune_node(self.root_, X_val.reset_index(drop=True), y_encoded, weights)

    def stats(self) -> dict[str, int]:
        self._check_is_fitted()
        return {
            "depth": self._depth(self.root_),
            "nodes": self._n_nodes(self.root_),
            "leaves": self._n_leaves(self.root_),
        }

    def export_text(self, max_depth: int = 4) -> str:
        self._check_is_fitted()
        lines: list[str] = []
        self._export_node(self.root_, lines, max_depth=max_depth)
        return "\n".join(lines)

    def _build_node(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
    ) -> Node:
        node = self._make_leaf(y, weights, depth)
        impurity = self._gini(node.class_counts)

        if (
            depth >= self.max_depth
            or weights.sum() < self.min_samples_split
            or impurity <= 1e-12
        ):
            return node

        split = self._find_best_split(X, y, weights, impurity)
        if split is None or split.gain < self.min_gain:
            return node

        left_data, right_data = self._materialize_split(X, y, weights, split)
        X_left, y_left, w_left = left_data
        X_right, y_right, w_right = right_data

        if w_left.sum() < self.min_samples_leaf or w_right.sum() < self.min_samples_leaf:
            return node

        node.is_leaf = False
        node.feature = split.feature
        node.feature_type = split.feature_type
        node.threshold = split.threshold
        node.category = split.category
        node.gain = split.gain
        node.p_left = split.p_left
        node.p_right = split.p_right
        node.left = self._build_node(X_left, y_left, w_left, depth + 1)
        node.right = self._build_node(X_right, y_right, w_right, depth + 1)
        return node

    def _find_best_split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        parent_impurity: float,
    ) -> Split | None:
        best: Split | None = None
        for feature in X.columns:
            feature_type = self.feature_types_[feature]
            if feature_type == "numeric":
                for threshold in self._numeric_thresholds(X[feature]):
                    split = self._evaluate_numeric_split(
                        X[feature], y, weights, parent_impurity, feature, threshold
                    )
                    if split is not None and (best is None or split.gain > best.gain):
                        best = split
            elif feature_type == "categorical":
                categories = pd.Series(X[feature].dropna().unique()).sort_values().tolist()
                for category in categories:
                    split = self._evaluate_categorical_split(
                        X[feature], y, weights, parent_impurity, feature, category
                    )
                    if split is not None and (best is None or split.gain > best.gain):
                        best = split
            else:
                raise ValueError(f"Unknown feature type for {feature}: {feature_type}")
        return best

    def _numeric_thresholds(self, column: pd.Series) -> np.ndarray:
        values = pd.to_numeric(column, errors="coerce").dropna().to_numpy(dtype=float)
        unique_values = np.unique(values)
        if len(unique_values) < 2:
            return np.array([])
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
        if len(thresholds) <= self.max_thresholds:
            return thresholds
        indexes = np.linspace(0, len(thresholds) - 1, self.max_thresholds).astype(int)
        return thresholds[indexes]

    def _evaluate_numeric_split(
        self,
        column: pd.Series,
        y: np.ndarray,
        weights: np.ndarray,
        parent_impurity: float,
        feature: str,
        threshold: float,
    ) -> Split | None:
        known = column.notna().to_numpy()
        values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
        left_known = known & (values <= threshold)
        right_known = known & (values > threshold)
        return self._evaluate_known_masks(
            y, weights, parent_impurity, feature, "numeric", left_known, right_known, threshold, None
        )

    def _evaluate_categorical_split(
        self,
        column: pd.Series,
        y: np.ndarray,
        weights: np.ndarray,
        parent_impurity: float,
        feature: str,
        category: Any,
    ) -> Split | None:
        known = column.notna().to_numpy()
        left_known = known & (column.to_numpy() == category)
        right_known = known & (column.to_numpy() != category)
        return self._evaluate_known_masks(
            y, weights, parent_impity=parent_impurity, feature=feature,
            feature_type="categorical", left_known=left_known, right_known=right_known,
            threshold=None, category=category
        )

    def _evaluate_known_masks(
        self,
        y: np.ndarray,
        weights: np.ndarray,
        parent_impity: float,
        feature: str,
        feature_type: str,
        left_known: np.ndarray,
        right_known: np.ndarray,
        threshold: float | None,
        category: Any | None,
    ) -> Split | None:
        missing = ~(left_known | right_known)
        known_weight = weights[left_known | right_known].sum()
        if known_weight <= 0.0:
            return None

        left_known_weight = weights[left_known].sum()
        right_known_weight = weights[right_known].sum()
        if left_known_weight <= 0.0 or right_known_weight <= 0.0:
            return None

        p_left = float(left_known_weight / known_weight)
        p_right = 1.0 - p_left
        left_weights = np.where(left_known, weights, 0.0) + np.where(missing, weights * p_left, 0.0)
        right_weights = np.where(right_known, weights, 0.0) + np.where(missing, weights * p_right, 0.0)

        left_total = left_weights.sum()
        right_total = right_weights.sum()
        total = weights.sum()
        if left_total < self.min_samples_leaf or right_total < self.min_samples_leaf:
            return None

        left_impurity = self._gini(self._weighted_counts(y, left_weights))
        right_impurity = self._gini(self._weighted_counts(y, right_weights))
        gain = parent_impity - (left_total / total) * left_impurity - (right_total / total) * right_impurity
        return Split(
            feature=feature,
            feature_type=feature_type,
            gain=float(gain),
            p_left=p_left,
            p_right=p_right,
            threshold=threshold,
            category=category,
        )

    def _materialize_split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        split: Split,
    ) -> tuple[tuple[pd.DataFrame, np.ndarray, np.ndarray], tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
        left_known, right_known, missing = self._split_masks(X[split.feature], split)

        left_mask = left_known | missing
        right_mask = right_known | missing

        X_left = X.loc[left_mask].reset_index(drop=True)
        y_left = y[left_mask]
        w_left = weights[left_mask].copy()
        w_left[missing[left_mask]] *= split.p_left

        X_right = X.loc[right_mask].reset_index(drop=True)
        y_right = y[right_mask]
        w_right = weights[right_mask].copy()
        w_right[missing[right_mask]] *= split.p_right

        return (X_left, y_left, w_left), (X_right, y_right, w_right)

    def _split_masks(self, column: pd.Series, split: Split) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        known = column.notna().to_numpy()
        if split.feature_type == "numeric":
            values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
            left_known = known & (values <= split.threshold)
            right_known = known & (values > split.threshold)
        else:
            values = column.to_numpy()
            left_known = known & (values == split.category)
            right_known = known & (values != split.category)
        missing = ~known
        return left_known, right_known, missing

    def _predict_row_proba(self, row: pd.Series, node: Node) -> np.ndarray:
        if node.is_leaf:
            return node.probabilities

        value = row[node.feature]
        if pd.isna(value):
            left_proba = self._predict_row_proba(row, node.left)
            right_proba = self._predict_row_proba(row, node.right)
            return node.p_left * left_proba + node.p_right * right_proba

        if node.feature_type == "numeric":
            go_left = float(value) <= node.threshold
        else:
            go_left = value == node.category
        return self._predict_row_proba(row, node.left if go_left else node.right)

    def _prune_node(self, node: Node, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> int:
        if node.is_leaf or len(X) == 0:
            return 0

        left_data, right_data = self._route_validation(X, y, weights, node)
        pruned = self._prune_node(node.left, *left_data)
        pruned += self._prune_node(node.right, *right_data)

        subtree_predictions = np.argmax(
            np.vstack([self._predict_row_proba(row, node) for _, row in X.iterrows()]),
            axis=1,
        )
        subtree_error = weights[subtree_predictions != y].sum()
        leaf_error = weights[node.prediction_index != y].sum()

        if leaf_error <= subtree_error + self.prune_tolerance:
            node.is_leaf = True
            node.feature = None
            node.feature_type = None
            node.threshold = None
            node.category = None
            node.left = None
            node.right = None
            return pruned + 1
        return pruned

    def _route_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        node: Node,
    ) -> tuple[tuple[pd.DataFrame, np.ndarray, np.ndarray], tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
        split = Split(
            feature=node.feature,
            feature_type=node.feature_type,
            gain=node.gain,
            p_left=node.p_left,
            p_right=node.p_right,
            threshold=node.threshold,
            category=node.category,
        )
        left_known, right_known, missing = self._split_masks(X[node.feature], split)
        left_mask = left_known | missing
        right_mask = right_known | missing

        X_left = X.loc[left_mask].reset_index(drop=True)
        y_left = y[left_mask]
        w_left = weights[left_mask].copy()
        w_left[missing[left_mask]] *= node.p_left

        X_right = X.loc[right_mask].reset_index(drop=True)
        y_right = y[right_mask]
        w_right = weights[right_mask].copy()
        w_right[missing[right_mask]] *= node.p_right
        return (X_left, y_left, w_left), (X_right, y_right, w_right)

    def _make_leaf(self, y: np.ndarray, weights: np.ndarray, depth: int) -> Node:
        counts = self._weighted_counts(y, weights)
        total = counts.sum()
        probabilities = counts / total if total > 0 else np.ones(len(counts)) / len(counts)
        return Node(
            depth=depth,
            class_counts=counts,
            prediction_index=int(np.argmax(probabilities)),
            probabilities=probabilities,
            weight=float(total),
        )

    def _weighted_counts(self, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.bincount(y, weights=weights, minlength=len(self.classes_)).astype(float)

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        total = counts.sum()
        if total <= 0.0:
            return 0.0
        probabilities = counts / total
        return float(1.0 - np.sum(probabilities**2))

    def _encode_y(self, y: np.ndarray) -> np.ndarray:
        encoded = pd.Series(y).map(self.class_to_index_)
        if encoded.isna().any():
            raise ValueError("Target contains unknown labels.")
        return encoded.astype(int).to_numpy()

    def _check_is_fitted(self) -> None:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("Call fit before using the classifier.")

    def _depth(self, node: Node | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return node.depth
        return max(self._depth(node.left), self._depth(node.right))

    def _n_nodes(self, node: Node | None) -> int:
        if node is None:
            return 0
        return 1 + self._n_nodes(node.left) + self._n_nodes(node.right)

    def _n_leaves(self, node: Node | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._n_leaves(node.left) + self._n_leaves(node.right)

    def _export_node(self, node: Node, lines: list[str], max_depth: int) -> None:
        indent = "  " * node.depth
        predicted_class = self.classes_[node.prediction_index]
        if node.is_leaf or node.depth >= max_depth:
            lines.append(
                f"{indent}Leaf(class={predicted_class}, p={node.probabilities.round(3).tolist()}, "
                f"weight={node.weight:.1f})"
            )
            return

        if node.feature_type == "numeric":
            condition = f"{node.feature} <= {node.threshold:.3f}"
        else:
            condition = f"{node.feature} == {node.category!r}"
        lines.append(
            f"{indent}Node({condition}, gain={node.gain:.4f}, "
            f"missing=({node.p_left:.2f}, {node.p_right:.2f}))"
        )
        self._export_node(node.left, lines, max_depth)
        self._export_node(node.right, lines, max_depth)
