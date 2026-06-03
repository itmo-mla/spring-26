from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Split:
    feature: str
    feature_type: str
    gain: float
    p_left: float
    p_right: float
    threshold: float | None = None
    category: Any | None = None


@dataclass
class TreeNode:
    node_id: int
    depth: int
    class_counts: np.ndarray
    prediction: int
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
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class ProbabilisticID3Classifier:
    """Binary ID3 tree with Gini criterion and probabilistic missing routing."""

    def __init__(
        self,
        max_depth: int = 7,
        min_samples_split: int = 40,
        min_samples_leaf: int = 15,
        min_gain: float = 1e-4,
        max_thresholds: int = 48,
        prune_tolerance: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_thresholds = max_thresholds
        self.prune_tolerance = prune_tolerance
        self.root_: TreeNode | None = None
        self.classes_: np.ndarray | None = None
        self.class_to_index_: dict[Any, int] = {}
        self.feature_types_: dict[str, str] = {}
        self.feature_importances_: pd.Series | None = None
        self._next_node_id = 0
        self._raw_importances: dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_types: dict[str, str],
    ) -> "ProbabilisticID3Classifier":
        X = X.reset_index(drop=True).copy()
        y_encoded = self._fit_classes(y)
        weights = np.ones(len(y_encoded), dtype=float)

        self.feature_types_ = dict(feature_types)
        self._next_node_id = 0
        self._raw_importances = {feature: 0.0 for feature in X.columns}
        self.root_ = self._build_node(X, y_encoded, weights, depth=0)
        self._normalize_importances()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        X = X.reset_index(drop=True)
        return np.vstack([self._predict_row_proba(row, self.root_) for _, row in X.iterrows()])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        class_indexes = np.argmax(probabilities, axis=1)
        return self.classes_[class_indexes]

    def prune(self, X_val: pd.DataFrame, y_val: np.ndarray) -> int:
        self._check_is_fitted()
        y_encoded = self._encode_y(y_val)
        weights = np.ones(len(y_encoded), dtype=float)
        return self._prune_node(self.root_, X_val.reset_index(drop=True), y_encoded, weights)

    def stats(self) -> dict[str, int]:
        self._check_is_fitted()
        return {
            "depth": self.depth(),
            "nodes": self.n_nodes(),
            "leaves": self.n_leaves(),
        }

    def depth(self) -> int:
        return self._depth(self.root_)

    def n_nodes(self) -> int:
        return self._n_nodes(self.root_)

    def n_leaves(self) -> int:
        return self._n_leaves(self.root_)

    def export_text(self, max_depth: int = 4) -> str:
        self._check_is_fitted()
        lines: list[str] = []
        self._export_node(self.root_, lines, max_depth=max_depth)
        return "\n".join(lines)

    def _fit_classes(self, y: np.ndarray) -> np.ndarray:
        self.classes_ = np.sort(pd.Series(y).unique())
        self.class_to_index_ = {label: idx for idx, label in enumerate(self.classes_)}
        return self._encode_y(y)

    def _encode_y(self, y: np.ndarray) -> np.ndarray:
        series = pd.Series(y)
        encoded = series.map(self.class_to_index_)
        if encoded.isna().any():
            unknown = sorted(series[encoded.isna()].unique())
            raise ValueError(f"Unknown target labels: {unknown}")
        return encoded.astype(int).to_numpy()

    def _new_leaf(self, y: np.ndarray, weights: np.ndarray, depth: int) -> TreeNode:
        counts = self._weighted_counts(y, weights)
        total = counts.sum()
        probabilities = counts / total if total > 0 else np.ones(len(counts)) / len(counts)
        prediction = int(np.argmax(probabilities))
        node = TreeNode(
            node_id=self._next_node_id,
            depth=depth,
            class_counts=counts,
            prediction=prediction,
            probabilities=probabilities,
            weight=float(total),
        )
        self._next_node_id += 1
        return node

    def _build_node(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
    ) -> TreeNode:
        node = self._new_leaf(y, weights, depth)
        total_weight = weights.sum()
        impurity = self._gini(node.class_counts)

        if (
            depth >= self.max_depth
            or total_weight < self.min_samples_split
            or impurity <= 1e-12
        ):
            return node

        split = self._find_best_split(X, y, weights, parent_impurity=impurity)
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
        self._raw_importances[split.feature] += split.gain * total_weight

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
                candidates = self._numeric_thresholds(X[feature])
                for threshold in candidates:
                    split = self._evaluate_numeric_split(
                        X[feature],
                        y,
                        weights,
                        parent_impurity,
                        feature,
                        float(threshold),
                    )
                    if split is not None and (best is None or split.gain > best.gain):
                        best = split
            elif feature_type == "categorical":
                categories = pd.Series(X[feature].dropna().unique()).sort_values().tolist()
                for category in categories:
                    split = self._evaluate_categorical_split(
                        X[feature],
                        y,
                        weights,
                        parent_impurity,
                        feature,
                        category,
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
            return np.array([], dtype=float)
        if len(unique_values) <= self.max_thresholds + 1:
            return (unique_values[:-1] + unique_values[1:]) / 2.0
        quantiles = np.linspace(0.0, 1.0, self.max_thresholds + 2)[1:-1]
        return np.unique(np.quantile(values, quantiles))

    def _evaluate_numeric_split(
        self,
        column: pd.Series,
        y: np.ndarray,
        weights: np.ndarray,
        parent_impurity: float,
        feature: str,
        threshold: float,
    ) -> Split | None:
        numeric = pd.to_numeric(column, errors="coerce")
        observed = ~numeric.isna().to_numpy()
        values = numeric.to_numpy(dtype=float)
        left = observed & (values <= threshold)
        right = observed & (values > threshold)
        result = self._score_binary_split(y, weights, left, right, parent_impurity)
        if result is None:
            return None
        gain, p_left, p_right = result
        return Split(
            feature=feature,
            feature_type="numeric",
            threshold=threshold,
            gain=gain,
            p_left=p_left,
            p_right=p_right,
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
        observed = ~column.isna().to_numpy()
        values = column.to_numpy(dtype=object)
        left = observed & (values == category)
        right = observed & (values != category)
        result = self._score_binary_split(y, weights, left, right, parent_impurity)
        if result is None:
            return None
        gain, p_left, p_right = result
        return Split(
            feature=feature,
            feature_type="categorical",
            category=category,
            gain=gain,
            p_left=p_left,
            p_right=p_right,
        )

    def _score_binary_split(
        self,
        y: np.ndarray,
        weights: np.ndarray,
        left_observed: np.ndarray,
        right_observed: np.ndarray,
        parent_impurity: float,
    ) -> tuple[float, float, float] | None:
        left_observed_weight = weights[left_observed].sum()
        right_observed_weight = weights[right_observed].sum()
        observed_weight = left_observed_weight + right_observed_weight
        if left_observed_weight <= 0 or right_observed_weight <= 0 or observed_weight <= 0:
            return None

        p_left = left_observed_weight / observed_weight
        p_right = right_observed_weight / observed_weight
        missing = ~(left_observed | right_observed)

        left_counts = self._weighted_counts(y[left_observed], weights[left_observed])
        right_counts = self._weighted_counts(y[right_observed], weights[right_observed])
        if missing.any():
            missing_counts = self._weighted_counts(y[missing], weights[missing])
            left_counts = left_counts + p_left * missing_counts
            right_counts = right_counts + p_right * missing_counts

        left_weight = left_counts.sum()
        right_weight = right_counts.sum()
        total_weight = left_weight + right_weight
        if left_weight < self.min_samples_leaf or right_weight < self.min_samples_leaf:
            return None

        child_impurity = (
            (left_weight / total_weight) * self._gini(left_counts)
            + (right_weight / total_weight) * self._gini(right_counts)
        )
        return parent_impurity - child_impurity, float(p_left), float(p_right)

    def _materialize_split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        split: Split | TreeNode,
    ) -> tuple[tuple[pd.DataFrame, np.ndarray, np.ndarray], tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
        if split.feature_type == "numeric":
            numeric = pd.to_numeric(X[split.feature], errors="coerce")
            observed = ~numeric.isna().to_numpy()
            values = numeric.to_numpy(dtype=float)
            left_mask = observed & (values <= split.threshold)
            right_mask = observed & (values > split.threshold)
        else:
            column = X[split.feature]
            observed = ~column.isna().to_numpy()
            values = column.to_numpy(dtype=object)
            left_mask = observed & (values == split.category)
            right_mask = observed & (values != split.category)

        missing_mask = ~(left_mask | right_mask)
        left = self._weighted_subset(X, y, weights, left_mask, missing_mask, split.p_left)
        right = self._weighted_subset(X, y, weights, right_mask, missing_mask, split.p_right)
        return left, right

    @staticmethod
    def _weighted_subset(
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        observed_mask: np.ndarray,
        missing_mask: np.ndarray,
        missing_fraction: float,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        frames = [X.iloc[observed_mask].copy()]
        targets = [y[observed_mask]]
        branch_weights = [weights[observed_mask]]

        if missing_mask.any() and missing_fraction > 0:
            frames.append(X.iloc[missing_mask].copy())
            targets.append(y[missing_mask])
            branch_weights.append(weights[missing_mask] * missing_fraction)

        return (
            pd.concat(frames, axis=0, ignore_index=True),
            np.concatenate(targets),
            np.concatenate(branch_weights),
        )

    def _prune_node(
        self,
        node: TreeNode | None,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        weights: np.ndarray,
    ) -> int:
        if node is None or node.is_leaf or len(y_val) == 0 or weights.sum() <= 0:
            return 0

        left_data, right_data = self._materialize_split(X_val, y_val, weights, node)
        X_left, y_left, w_left = left_data
        X_right, y_right, w_right = right_data

        pruned = self._prune_node(node.left, X_left, y_left, w_left)
        pruned += self._prune_node(node.right, X_right, y_right, w_right)

        subtree_error = self._weighted_error_from_node(node, X_val, y_val, weights)
        leaf_prediction = np.full(len(y_val), node.prediction, dtype=int)
        leaf_error = self._weighted_error(leaf_prediction, y_val, weights)

        if leaf_error <= subtree_error + self.prune_tolerance:
            node.is_leaf = True
            node.feature = None
            node.feature_type = None
            node.threshold = None
            node.category = None
            node.left = None
            node.right = None
            node.gain = 0.0
            pruned += 1
        return pruned

    def _weighted_error_from_node(
        self,
        node: TreeNode,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        probabilities = np.vstack([self._predict_row_proba(row, node) for _, row in X.iterrows()])
        predictions = np.argmax(probabilities, axis=1)
        return self._weighted_error(predictions, y, weights)

    @staticmethod
    def _weighted_error(predictions: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        if weights.sum() <= 0:
            return 0.0
        return float(np.average(predictions != y, weights=weights))

    def _predict_row_proba(self, row: pd.Series, node: TreeNode | None) -> np.ndarray:
        if node is None:
            return np.ones(len(self.classes_)) / len(self.classes_)
        if node.is_leaf:
            return node.probabilities

        value = row[node.feature]
        if pd.isna(value):
            return (
                node.p_left * self._predict_row_proba(row, node.left)
                + node.p_right * self._predict_row_proba(row, node.right)
            )

        if node.feature_type == "numeric":
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric_value):
                return (
                    node.p_left * self._predict_row_proba(row, node.left)
                    + node.p_right * self._predict_row_proba(row, node.right)
                )
            if numeric_value <= node.threshold:
                return self._predict_row_proba(row, node.left)
            return self._predict_row_proba(row, node.right)

        if value == node.category:
            return self._predict_row_proba(row, node.left)
        return self._predict_row_proba(row, node.right)

    def _weighted_counts(self, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.bincount(y, weights=weights, minlength=len(self.classes_)).astype(float)

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        total = counts.sum()
        if total <= 0:
            return 0.0
        probabilities = counts / total
        return float(1.0 - np.dot(probabilities, probabilities))

    def _normalize_importances(self) -> None:
        values = pd.Series(self._raw_importances, dtype=float).sort_values(ascending=False)
        total = values.sum()
        self.feature_importances_ = values / total if total > 0 else values

    def _depth(self, node: TreeNode | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def _n_nodes(self, node: TreeNode | None) -> int:
        if node is None:
            return 0
        return 1 + self._n_nodes(node.left) + self._n_nodes(node.right)

    def _n_leaves(self, node: TreeNode | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._n_leaves(node.left) + self._n_leaves(node.right)

    def _export_node(self, node: TreeNode | None, lines: list[str], max_depth: int, prefix: str = "") -> None:
        if node is None:
            return

        class_name = self.classes_[node.prediction]
        confidence = node.probabilities[node.prediction]
        if node.is_leaf or node.depth >= max_depth:
            suffix = "..." if (not node.is_leaf and node.depth >= max_depth) else ""
            lines.append(
                f"{prefix}leaf{suffix}: class={class_name}, p={confidence:.3f}, weight={node.weight:.1f}"
            )
            return

        if node.feature_type == "numeric":
            rule = f"{node.feature} <= {node.threshold:.3f}"
        else:
            rule = f"{node.feature} == {node.category}"
        lines.append(
            f"{prefix}if {rule} "
            f"(gain={node.gain:.4f}, missing left/right={node.p_left:.2f}/{node.p_right:.2f})"
        )
        self._export_node(node.left, lines, max_depth=max_depth, prefix=prefix + "  yes -> ")
        self._export_node(node.right, lines, max_depth=max_depth, prefix=prefix + "  no  -> ")

    def _check_is_fitted(self) -> None:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("The tree is not fitted yet.")
