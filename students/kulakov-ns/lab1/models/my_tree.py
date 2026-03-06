from dataclasses import dataclass
from typing import Dict, Tuple, Self
import numpy as np
import pandas as pd


@dataclass
class Node:
    is_leaf: bool
    prediction: int
    class_probs: np.ndarray
    weight_sum: float
    depth: int
    feature: str = None
    feature_type: str = None
    threshold: float = None
    category: object = None
    p_left: float = None
    p_right: float = None
    left: Self | None = None
    right: Self | None = None

    def _predict_proba_from_node(self, X, zeros_array):
        idx = np.arange(len(X))
        path_w = np.ones(len(X), dtype=float)
        self._accumulate_proba(X.reset_index(drop=True), idx, path_w, zeros_array)
        row_sums = zeros_array.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return zeros_array / row_sums
    
    def _accumulate_proba(self, X, idx, path_w, out):
        if len(idx) == 0:
            return
        if self.is_leaf:
            out[idx] += path_w[:, None] * self.class_probs[None, :]
            return

        col = X.loc[idx, self.feature].reset_index(drop=True)

        if self.feature_type == "num":
            left_det = ((col <= self.threshold) & ~col.isna()).to_numpy()
            right_det = ((col > self.threshold) & ~col.isna()).to_numpy()
        else:
            left_det = ((col == self.category) & ~col.isna()).to_numpy()
            right_det = ((col != self.category) & ~col.isna()).to_numpy()

        miss = col.isna().to_numpy()

        if left_det.any():
            self.left._accumulate_proba(X, idx[left_det], path_w[left_det], out)
        if right_det.any():
            self.right._accumulate_proba(X, idx[right_det], path_w[right_det], out)
        if miss.any():
            if self.p_left > 0:
                self.left._accumulate_proba(
                 X, idx[miss], path_w[miss] * self.p_left, out
                )
            if self.p_right > 0:
                self.right._accumulate_proba(
                    X, idx[miss], path_w[miss] * self.p_right, out
                )


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=5,
        min_gain=1e-4,
        random_state=42,
    ):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.random_state = random_state
        self.root = None
        self.status: str = "not_fitted"

    def fit(self, X, y):
        X = X.reset_index(drop=True).copy()
        y = pd.Series(y).reset_index(drop=True)

        self.classes_ = np.sort(y.unique())
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        y_enc = y.map(self.class_to_idx_).astype(int).to_numpy()
        w = np.ones(len(y_enc), dtype=float)

        self.features_ = list(X.columns)
        self.n_classes_ = len(self.classes_)
        self.root = self._build_tree(X, y_enc, w, depth=0)

        self.status = "before"

        return self

    def predict(self, X):
        X = X.reset_index(drop=True).copy()
        probs = self.root._predict_proba_from_node(X, np.zeros((len(X), self.n_classes_), dtype=float))
        return self.classes_[np.argmax(probs, axis=1)]

    def depth(self):
        return self._depth(self.root)

    def n_nodes(self):
        return self._n_nodes(self.root)

    def n_leaves(self):
        return self._n_leaves(self.root)

    def _depth(self, node):
        if node is None or node.is_leaf:
            return 1 if node is not None else 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def _n_nodes(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self._n_nodes(node.left) + self._n_nodes(node.right)

    def _n_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._n_leaves(node.left) + self._n_leaves(node.right)

    def _build_tree(self, X, y, w, depth):
        counts = self._weighted_counts(y, w)
        probs = counts / counts.sum()
        prediction = int(np.argmax(probs))
        node = Node(
            is_leaf=True,
            prediction=prediction,
            class_probs=probs,
            weight_sum=float(w.sum()),
            depth=depth,
        )

        if (
            depth >= self.max_depth
            or w.sum() < self.min_samples_split
            or np.count_nonzero(counts) == 1
        ):
            return node

        best = self._find_best_split(X, y, w)
        if best is None or best["gain"] < self.min_gain:
            return node

        left_data, right_data = self._materialize_split(
            X,
            y,
            w,
            best["feature"],
            best["feature_type"],
            best.get("threshold"),
            best.get("category"),
            best["p_left"],
            best["p_right"],
        )

        X_left, y_left, w_left = left_data
        X_right, y_right, w_right = right_data

        if w_left.sum() < self.min_samples_leaf or w_right.sum() < self.min_samples_leaf:
            return node

        node.is_leaf = False
        node.feature = best["feature"]
        node.feature_type = best["feature_type"]
        node.threshold = best.get("threshold")
        node.category = best.get("category")
        node.p_left = best["p_left"]
        node.p_right = best["p_right"]
        node.left = self._build_tree(X_left, y_left, w_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, w_right, depth + 1)
        return node

    def _find_best_split(self, X, y, w):
        parent_gini = self._gini_from_counts(self._weighted_counts(y, w))
        best = None

        for feature in self.features_:
            feature_type = self.feature_types[feature]
            col = X[feature]

            if feature_type == "num":
                values = np.sort(col.dropna().unique())
                if len(values) < 2:
                    continue
                candidates = (values[:-1] + values[1:]) / 2.0
                for threshold in candidates:
                    res = self._evaluate_split(col, y, w, feature_type, threshold=threshold)
                    if res is None:
                        continue
                    gain = parent_gini - res["child_gini"]
                    if best is None or gain > best["gain"]:
                        best = {
                            "feature": feature,
                            "feature_type": feature_type,
                            "threshold": float(threshold),
                            "p_left": res["p_left"],
                            "p_right": res["p_right"],
                            "gain": float(gain),
                        }
            else:
                categories = pd.Series(col.dropna().unique()).tolist()
                if len(categories) < 2:
                    continue
                for category in categories:
                    res = self._evaluate_split(col, y, w, feature_type, category=category)
                    if res is None:
                        continue
                    gain = parent_gini - res["child_gini"]
                    if best is None or gain > best["gain"]:
                        best = {
                            "feature": feature,
                            "feature_type": feature_type,
                            "category": category,
                            "p_left": res["p_left"],
                            "p_right": res["p_right"],
                            "gain": float(gain),
                        }
        return best

    def _evaluate_split(self, col, y, w, feature_type, threshold=None, category=None):
        s = pd.Series(col)
        missing_mask = s.isna().to_numpy()

        if feature_type == "num":
            observed_left = ((s <= threshold) & ~s.isna()).to_numpy()
            observed_right = ((s > threshold) & ~s.isna()).to_numpy()
        else:
            observed_left = ((s == category) & ~s.isna()).to_numpy()
            observed_right = ((s != category) & ~s.isna()).to_numpy()

        w_left_obs = w[observed_left].sum()
        w_right_obs = w[observed_right].sum()
        observed_total = w_left_obs + w_right_obs

        if w_left_obs <= 0 or w_right_obs <= 0:
            return None

        p_left = w_left_obs / observed_total
        p_right = w_right_obs / observed_total

        miss_counts = self._weighted_counts(y[missing_mask], w[missing_mask])
        left_counts = self._weighted_counts(y[observed_left], w[observed_left]) + p_left * miss_counts
        right_counts = self._weighted_counts(y[observed_right], w[observed_right]) + p_right * miss_counts

        w_left = left_counts.sum()
        w_right = right_counts.sum()

        if w_left < self.min_samples_leaf or w_right < self.min_samples_leaf:
            return None

        child_gini = (
            (w_left / (w_left + w_right)) * self._gini_from_counts(left_counts)
            + (w_right / (w_left + w_right)) * self._gini_from_counts(right_counts)
        )

        return {"child_gini": child_gini, "p_left": p_left, "p_right": p_right}

    def _materialize_split(
        self, X, y, w, feature, feature_type, threshold=None, category=None, p_left=0.5, p_right=0.5
    ):
        s = X[feature]
        missing_mask = s.isna().to_numpy()

        if feature_type == "num":
            left_mask = ((s <= threshold) & ~s.isna()).to_numpy()
            right_mask = ((s > threshold) & ~s.isna()).to_numpy()
        else:
            left_mask = ((s == category) & ~s.isna()).to_numpy()
            right_mask = ((s != category) & ~s.isna()).to_numpy()

        X_left_parts = [X.loc[left_mask].copy()]
        y_left_parts = [y[left_mask]]
        w_left_parts = [w[left_mask]]

        X_right_parts = [X.loc[right_mask].copy()]
        y_right_parts = [y[right_mask]]
        w_right_parts = [w[right_mask]]

        if missing_mask.any():
            if p_left > 0:
                X_left_parts.append(X.loc[missing_mask].copy())
                y_left_parts.append(y[missing_mask])
                w_left_parts.append(w[missing_mask] * p_left)
            if p_right > 0:
                X_right_parts.append(X.loc[missing_mask].copy())
                y_right_parts.append(y[missing_mask])
                w_right_parts.append(w[missing_mask] * p_right)

        X_left = pd.concat(X_left_parts, axis=0, ignore_index=True)
        y_left = np.concatenate(y_left_parts)
        w_left = np.concatenate(w_left_parts)

        X_right = pd.concat(X_right_parts, axis=0, ignore_index=True)
        y_right = np.concatenate(y_right_parts)
        w_right = np.concatenate(w_right_parts)

        return (X_left, y_left, w_left), (X_right, y_right, w_right)


    def _weighted_counts(self, y, w):
        if len(y) == 0:
            return np.zeros(self.n_classes_, dtype=float)
        return np.bincount(y, weights=w, minlength=self.n_classes_).astype(float)

    def _gini_from_counts(self, counts):
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return 1.0 - np.sum(p ** 2)
    
    def get_stats(self) -> Dict[str, int]:
        return {
            f"{self.status}_depth": self.depth(),
            f"{self.status}_nodes": self.n_nodes(),
            f"{self.status}_leaves": self.n_leaves(),
        }
    

class Pruner:
    def reduced_error_prune(self, tree: DecisionTree, X_val, y_val) -> DecisionTree:
        X_val = X_val.reset_index(drop=True).copy()
        y_val = pd.Series(y_val).reset_index(drop=True)
        y_enc = y_val.map(tree.class_to_idx_).astype(int).to_numpy()
        w = np.ones(len(y_enc), dtype=float)
        self._prune(tree, tree.root, X_val, y_enc, w)
        tree.status = "after"
        return tree

    def _prune(self, tree: DecisionTree, node, X, y, w):
        if node is None or node.is_leaf or len(X) == 0:
            return

        left_data, right_data = tree._materialize_split(
            X,
            y,
            w,
            node.feature,
            node.feature_type,
            node.threshold,
            node.category,
            node.p_left,
            node.p_right,
        )

        X_left, y_left, w_left = left_data
        X_right, y_right, w_right = right_data

        self._prune(tree, node.left, X_left, y_left, w_left)
        self._prune(tree, node.right, X_right, y_right, w_right)

        subtree_probs = tree.root._predict_proba_from_node(X, np.zeros((len(X), tree.n_classes_), dtype=float))
        subtree_pred = np.argmax(subtree_probs, axis=1)
        subtree_acc = np.average(subtree_pred == y, weights=w)

        leaf_pred = np.full(len(y), node.prediction)
        leaf_acc = np.average(leaf_pred == y, weights=w)

        if leaf_acc >= subtree_acc:
            node.is_leaf = True
            node.feature = None
            node.feature_type = None
            node.threshold = None
            node.category = None
            node.p_left = None
            node.p_right = None
            node.left = None
            node.right = None
    