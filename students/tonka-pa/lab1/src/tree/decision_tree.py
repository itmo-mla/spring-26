from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from src.tree.tree_node import TreeNode


# ===========================================================================#
# ===========================================================================#


@dataclass
class CostComplexityPruningPath:
    """Класс для сохранения путей cost-complexity pruning."""

    ccp_alphas: np.ndarray
    impurities: np.ndarray
    node_counts: np.ndarray
    depths: np.ndarray
    trees: list[TreeNode] | None = None


@dataclass
class SplitCandidate:
    """Класс лучшего кандидата для сплита."""

    feature_index: int
    feature_name: str
    feature_type: str
    gain: float
    q_left: float
    q_right: float
    n_known: int
    n_left: int
    n_right: int
    left_mask: np.ndarray
    right_mask: np.ndarray
    threshold: float | None = None
    category: Any = None


# ===========================================================================#
# ===========================================================================#


class BaseDecisionTree(BaseEstimator):
    """Базовый класс дерева. Расширяется для классификации и регрессии."""

    def __init__(
        self,
        task: str,
        criterion: str,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> BaseDecisionTree:  # noqa: F821
        X_frame = self._as_frame(X)
        y_array = self._prepare_target(y)
        if len(X_frame) != len(y_array):
            raise ValueError("X and y must have the same length.")
        if len(y_array) == 0:
            raise ValueError("Cannot fit a tree on an empty dataset.")

        self.n_features_in_ = X_frame.shape[1]
        self.feature_names_in_ = np.asarray(X_frame.columns, dtype=object)
        self.feature_types_ = self._infer_feature_types(X_frame)
        self._feature_importance_raw_ = np.zeros(self.n_features_in_, dtype=float)

        self.root_ = self._build_tree(X_frame.reset_index(drop=True), y_array, depth=0)
        if self.ccp_alpha > 0:
            self._apply_ccp_alpha(float(self.ccp_alpha))
        self.feature_importances_ = self._compute_feature_importances()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        X_frame = self._as_frame(X)
        if self.task == "classification":
            proba = self.predict_proba(X_frame)
            return self.classes_[np.argmax(proba, axis=1)]
        return np.asarray(
            [self._predict_row_value(row, self.root_) for _, row in X_frame.iterrows()],
            dtype=float,
        )

    def cost_complexity_pruning_path(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> CostComplexityPruningPath:
        """Возвращает weakest-link pruning path для дерева, обученного на входных данных."""
        params = self.get_params()
        params["ccp_alpha"] = 0.0
        model = self.__class__(**params)
        model.fit(X, y)
        return model._pruning_path_from_root(model.root_, include_trees=False)

    def get_depth(self) -> int:
        self._check_is_fitted()
        return self._depth(self.root_)

    def get_n_leaves(self) -> int:
        self._check_is_fitted()
        return self._count_leaves(self.root_)

    def get_node_count(self) -> int:
        self._check_is_fitted()
        return self._count_nodes(self.root_)

    def export_text(self, max_depth: int | None = None) -> str:
        """Для текстового отображения дерева (не воспользовался)"""
        self._check_is_fitted()
        lines: list[str] = []

        def walk(node: TreeNode, indent: str) -> None:
            if max_depth is not None and node.depth > max_depth:
                lines.append(f"{indent}...")
                return
            if node.is_leaf:
                lines.append(
                    f"{indent}Leaf(n={node.n_samples}, impurity={node.impurity:.4f}, "
                    f"prediction={node.prediction!r})"
                )
                return
            lines.append(
                f"{indent}if {node.split_label()} "
                f"(gain={node.gain:.4f}, q_left={node.q_left:.3f}):"
            )
            walk(node.left, indent + "  ")
            lines.append(f"{indent}else:")
            walk(node.right, indent + "  ")

        walk(self.root_, "")
        return "\n".join(lines)

    # Дальше вспомогательные функции и функция построения самого дерева

    def _prepare_target(self, y: pd.Series | np.ndarray) -> np.ndarray:
        y_series = pd.Series(y).reset_index(drop=True)
        if self.task == "classification":
            self.classes_, encoded = np.unique(y_series.to_numpy(), return_inverse=True)
            self.n_classes_ = len(self.classes_)
            return encoded.astype(int)
        return pd.to_numeric(y_series, errors="raise").to_numpy(dtype=float)

    def _as_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if (
            hasattr(self, "feature_names_in_")
            and len(self.feature_names_in_) == X.shape[1]
        ):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame(X)

    def _infer_feature_types(self, X: pd.DataFrame) -> list[str]:
        types: list[str] = []
        for column in X.columns:
            series = X[column]
            if pd.api.types.is_bool_dtype(series):
                types.append("categorical")
            elif pd.api.types.is_numeric_dtype(series):
                types.append("numeric")
            else:
                types.append("categorical")
        return types

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int) -> TreeNode:
        """Рекурсивное построение дерева."""
        impurity = self._impurity(y)
        prediction, class_counts, proba = self._leaf_values(y)
        node = TreeNode(
            depth=depth,
            n_samples=len(y),
            impurity=impurity,
            prediction=prediction,
            class_counts=class_counts,
            proba=proba,
            leaf_risk=impurity * len(y),
        )

        if self._should_stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        min_gain = max(float(self.min_impurity_decrease), 1e-12)
        if split.gain <= min_gain:
            return node

        node.feature_index = split.feature_index
        node.feature_name = split.feature_name
        node.feature_type = split.feature_type
        node.threshold = split.threshold
        node.category = split.category
        node.gain = split.gain
        node.q_left = split.q_left
        node.q_right = split.q_right
        node.n_known = split.n_known
        node.n_left = split.n_left
        node.n_right = split.n_right
        node.left = self._build_tree(
            X.loc[split.left_mask].reset_index(drop=True),
            y[split.left_mask],
            depth + 1,
        )
        node.right = self._build_tree(
            X.loc[split.right_mask].reset_index(drop=True),
            y[split.right_mask],
            depth + 1,
        )
        return node

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if self.task == "classification":
            return len(np.unique(y)) == 1
        return bool(np.allclose(y, y[0]))

    def _best_split(self, X: pd.DataFrame, y: np.ndarray) -> SplitCandidate | None:
        best: SplitCandidate | None = None
        for feature_index, feature_name in enumerate(X.columns):
            feature_type = self.feature_types_[feature_index]
            series = X[feature_name]
            known_mask = series.notna().to_numpy()
            n_known = int(known_mask.sum())
            if n_known < 2 * self.min_samples_leaf:
                continue
            y_known = y[known_mask]
            parent_impurity = self._impurity(y_known)

            if feature_type == "numeric":
                candidates = self._numeric_candidates(
                    feature_index,
                    str(feature_name),
                    series,
                    y,
                    known_mask,
                    n_known,
                    parent_impurity,
                )
            else:
                candidates = self._categorical_candidates(
                    feature_index,
                    str(feature_name),
                    series,
                    y,
                    known_mask,
                    n_known,
                    parent_impurity,
                )

            for candidate in candidates:
                if best is None or candidate.gain > best.gain:
                    best = candidate
        return best

    def _numeric_candidates(
        self,
        feature_index: int,
        feature_name: str,
        series: pd.Series,
        y: np.ndarray,
        known_mask: np.ndarray,
        n_known: int,
        parent_impurity: float,
    ) -> list[SplitCandidate]:
        values = series.loc[known_mask].to_numpy(dtype=float)
        unique_values = np.unique(values)
        if len(unique_values) < 2:
            return []
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
        candidates: list[SplitCandidate] = []
        for threshold in thresholds:
            left_known = values <= threshold
            candidate = self._candidate_from_known_mask(
                feature_index=feature_index,
                feature_name=feature_name,
                feature_type="numeric",
                y=y,
                known_mask=known_mask,
                left_known=left_known,
                n_known=n_known,
                parent_impurity=parent_impurity,
                threshold=float(threshold),
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _categorical_candidates(
        self,
        feature_index: int,
        feature_name: str,
        series: pd.Series,
        y: np.ndarray,
        known_mask: np.ndarray,
        n_known: int,
        parent_impurity: float,
    ) -> list[SplitCandidate]:
        values = series.loc[known_mask].to_numpy()
        categories = pd.unique(values)
        if len(categories) < 2:
            return []
        candidates: list[SplitCandidate] = []
        for category in categories:
            left_known = values == category
            candidate = self._candidate_from_known_mask(
                feature_index=feature_index,
                feature_name=feature_name,
                feature_type="categorical",
                y=y,
                known_mask=known_mask,
                left_known=left_known,
                n_known=n_known,
                parent_impurity=parent_impurity,
                category=category,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _candidate_from_known_mask(
        self,
        feature_index: int,
        feature_name: str,
        feature_type: str,
        y: np.ndarray,
        known_mask: np.ndarray,
        left_known: np.ndarray,
        n_known: int,
        parent_impurity: float,
        threshold: float | None = None,
        category: Any = None,
    ) -> SplitCandidate | None:
        n_left = int(left_known.sum())
        n_right = n_known - n_left
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return None
        left_mask = np.zeros(len(y), dtype=bool)
        right_mask = np.zeros(len(y), dtype=bool)
        left_mask[known_mask] = left_known
        right_mask[known_mask] = ~left_known
        left_impurity = self._impurity(y[left_mask])
        right_impurity = self._impurity(y[right_mask])
        weighted_impurity = (
            n_left / n_known * left_impurity + n_right / n_known * right_impurity
        )
        gain = parent_impurity - weighted_impurity
        return SplitCandidate(
            feature_index=feature_index,
            feature_name=feature_name,
            feature_type=feature_type,
            gain=float(gain),
            q_left=n_left / n_known,
            q_right=n_right / n_known,
            n_known=n_known,
            n_left=n_left,
            n_right=n_right,
            left_mask=left_mask,
            right_mask=right_mask,
            threshold=threshold,
            category=category,
        )

    def _impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            counts = np.bincount(y.astype(int), minlength=self.n_classes_)
            probabilities = counts[counts > 0] / len(y)
            if self.criterion == "gini":
                return float(1.0 - np.sum(probabilities**2))
            if self.criterion == "entropy":
                return float(-np.sum(probabilities * np.log2(probabilities)))
            if self.criterion == "misclassification":
                return float(1.0 - np.max(probabilities))
            raise ValueError(f"Unsupported classification criterion: {self.criterion}")
        if self.criterion != "squared_error":
            raise ValueError(f"Unsupported regression criterion: {self.criterion}")
        return float(np.mean((y - np.mean(y)) ** 2))

    def _leaf_values(
        self,
        y: np.ndarray,
    ) -> tuple[Any, np.ndarray | None, np.ndarray | None]:
        if self.task == "classification":
            counts = np.bincount(y.astype(int), minlength=self.n_classes_)
            proba = counts / counts.sum()
            prediction_index = int(np.argmax(counts))
            return self.classes_[prediction_index], counts, proba
        return float(np.mean(y)), None, None

    def _predict_row_proba(self, row: pd.Series, node: TreeNode) -> np.ndarray:
        if node.is_leaf:
            return node.proba.copy()
        value = row.iloc[node.feature_index]
        if pd.isna(value):
            return node.q_left * self._predict_row_proba(
                row, node.left
            ) + node.q_right * self._predict_row_proba(row, node.right)
        if self._go_left(value, node):
            return self._predict_row_proba(row, node.left)
        return self._predict_row_proba(row, node.right)

    def _predict_row_value(self, row: pd.Series, node: TreeNode) -> float:
        if node.is_leaf:
            return float(node.prediction)
        value = row.iloc[node.feature_index]
        if pd.isna(value):
            return node.q_left * self._predict_row_value(
                row, node.left
            ) + node.q_right * self._predict_row_value(row, node.right)
        if self._go_left(value, node):
            return self._predict_row_value(row, node.left)
        return self._predict_row_value(row, node.right)

    def _go_left(self, value: Any, node: TreeNode) -> bool:
        if node.feature_type == "numeric":
            return float(value) <= float(node.threshold)
        return value == node.category

    def _apply_ccp_alpha(self, ccp_alpha: float) -> None:
        path = self._pruning_path_from_root(self.root_, include_trees=True)
        valid_indexes = np.flatnonzero(path.ccp_alphas <= ccp_alpha)
        index = int(valid_indexes[-1]) if len(valid_indexes) else 0
        self.root_ = copy.deepcopy(path.trees[index])

    def _pruning_path_from_root(
        self,
        root: TreeNode,
        include_trees: bool,
    ) -> CostComplexityPruningPath:
        current_root = copy.deepcopy(root)
        alphas = [0.0]
        impurities = [self._subtree_risk(current_root)]
        node_counts = [self._count_nodes(current_root)]
        depths = [self._depth(current_root)]
        trees = [copy.deepcopy(current_root)] if include_trees else None

        while not current_root.is_leaf:
            candidates = self._internal_nodes(current_root)
            weakest = min(candidates, key=self._effective_alpha)
            alpha = max(0.0, float(self._effective_alpha(weakest)))
            weakest.prune_to_leaf()
            alphas.append(alpha)
            impurities.append(self._subtree_risk(current_root))
            node_counts.append(self._count_nodes(current_root))
            depths.append(self._depth(current_root))
            if include_trees:
                trees.append(copy.deepcopy(current_root))

        return CostComplexityPruningPath(
            ccp_alphas=np.asarray(alphas, dtype=float),
            impurities=np.asarray(impurities, dtype=float),
            node_counts=np.asarray(node_counts, dtype=int),
            depths=np.asarray(depths, dtype=int),
            trees=trees,
        )

    def _effective_alpha(self, node: TreeNode) -> float:
        leaves = self._count_leaves(node)
        if leaves <= 1:
            return float("inf")
        return (node.leaf_risk - self._subtree_risk(node)) / (leaves - 1)

    def _subtree_risk(self, node: TreeNode) -> float:
        if node.is_leaf:
            return float(node.leaf_risk)
        return self._subtree_risk(node.left) + self._subtree_risk(node.right)

    def _internal_nodes(self, node: TreeNode) -> list[TreeNode]:
        if node.is_leaf:
            return []
        return [
            node,
            *self._internal_nodes(node.left),
            *self._internal_nodes(node.right),
        ]

    def _depth(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def _count_leaves(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _count_nodes(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def _compute_feature_importances(self) -> np.ndarray:
        importances = np.zeros(self.n_features_in_, dtype=float)

        def walk(node: TreeNode) -> None:
            if node.is_leaf:
                return
            importances[node.feature_index] += max(0.0, node.gain) * max(
                1, node.n_known
            )
            walk(node.left)
            walk(node.right)

        walk(self.root_)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "root_"):
            raise ValueError("The tree is not fitted yet.")


# ===========================================================================#
# ===========================================================================#


class MyDecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    """Классификационное дерево."""

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            task="classification",
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        self._check_is_fitted()
        X_frame = self._as_frame(X)
        return np.vstack(
            [self._predict_row_proba(row, self.root_) for _, row in X_frame.iterrows()]
        )


class MyDecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    """Регрессионное дерево."""

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            task="regression",
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )
