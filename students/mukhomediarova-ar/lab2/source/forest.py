from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class OOBRandomForestClassifier(ClassifierMixin, BaseEstimator):
    """Random Forest with explicit bootstrap, random subspaces and OOB estimates."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str | int | float | None = "sqrt",
        max_depth: int | None = None,
        max_leaf_nodes: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        bootstrap: bool = True,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "OOBRandomForestClassifier":
        X_frame = pd.DataFrame(X).reset_index(drop=True)
        y_array = np.asarray(y)
        X_array = X_frame.to_numpy(dtype=float)

        self.classes_ = np.sort(np.unique(y_array))
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_array.shape[1]
        self.feature_names_in_ = np.asarray(X_frame.columns, dtype=object)
        self.estimators_: list[DecisionTreeClassifier] = []
        self.feature_indices_: list[np.ndarray] = []
        self.bootstrap_indices_: list[np.ndarray] = []
        self.oob_indices_: list[np.ndarray] = []

        rng = np.random.RandomState(self.random_state)
        sample_indices = np.arange(len(X_array))

        for _ in range(self.n_estimators):
            tree_seed = int(rng.randint(0, np.iinfo(np.int32).max))
            if self.bootstrap:
                bootstrap_idx = rng.choice(sample_indices, size=len(sample_indices), replace=True)
            else:
                bootstrap_idx = sample_indices.copy()

            used = np.zeros(len(sample_indices), dtype=bool)
            used[bootstrap_idx] = True
            oob_idx = sample_indices[~used]

            feature_idx = np.sort(
                rng.choice(
                    self.n_features_in_,
                    size=self._resolve_max_features(self.n_features_in_),
                    replace=False,
                )
            )
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
            )
            tree.fit(X_array[bootstrap_idx][:, feature_idx], y_array[bootstrap_idx])

            self.estimators_.append(tree)
            self.feature_indices_.append(feature_idx)
            self.bootstrap_indices_.append(bootstrap_idx)
            self.oob_indices_.append(oob_idx)

        self.oob_decision_function_, self.oob_counts_ = self._collect_oob_probabilities(X_array)
        self.oob_score_ = self._score_oob_predictions(y_array)
        self.feature_importances_ = self._mean_impurity_importances()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_array = pd.DataFrame(X).to_numpy(dtype=float)
        probabilities = np.zeros((len(X_array), self.n_classes_), dtype=float)

        for tree, feature_idx in zip(self.estimators_, self.feature_indices_):
            probabilities += self._aligned_tree_proba(tree, X_array[:, feature_idx])

        return probabilities / len(self.estimators_)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: Any, y: Any) -> float:
        return float(accuracy_score(y, self.predict(X)))

    def compute_oob_permutation_importance(
        self,
        X: Any,
        y: Any,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Compute OOB^j importance as an OOB accuracy decrease after feature permutation."""
        self._check_fitted()
        X_array = pd.DataFrame(X).to_numpy(dtype=float)
        y_array = np.asarray(y)
        rng = np.random.RandomState(self.random_state if random_state is None else random_state)
        baseline = self.oob_score_
        rows: list[dict[str, float | str]] = []

        for feature_index, feature_name in enumerate(self.feature_names_in_):
            drops = []
            for _ in range(n_repeats):
                probabilities, counts = self._collect_oob_probabilities(
                    X_array,
                    permuted_feature=feature_index,
                    rng=rng,
                )
                mask = counts > 0
                predictions = self.classes_[np.argmax(probabilities[mask], axis=1)]
                permuted_score = accuracy_score(y_array[mask], predictions)
                drops.append(float(baseline - permuted_score))

            rows.append(
                {
                    "feature": str(feature_name),
                    "importance_mean": float(np.mean(drops)),
                    "importance_std": float(np.std(drops)),
                }
            )

        result = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        positive = result["importance_mean"].clip(lower=0.0)
        total = positive.sum()
        result["importance_normalized"] = positive / total if total > 0 else positive
        self.oob_permutation_importances_ = result
        return result

    def _resolve_max_features(self, n_features: int) -> int:
        value = self.max_features
        if value is None:
            return n_features
        if value == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if value == "log2":
            return max(1, int(math.log2(n_features)))
        if isinstance(value, float):
            if not 0 < value <= 1:
                raise ValueError("float max_features must be in (0, 1].")
            return max(1, int(math.ceil(value * n_features)))
        if isinstance(value, int):
            if not 1 <= value <= n_features:
                raise ValueError("int max_features must be in [1, n_features].")
            return value
        raise ValueError(f"Unsupported max_features value: {value!r}")

    def _collect_oob_probabilities(
        self,
        X_array: np.ndarray,
        permuted_feature: int | None = None,
        rng: np.random.RandomState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = np.zeros((len(X_array), self.n_classes_), dtype=float)
        counts = np.zeros(len(X_array), dtype=int)

        for tree, feature_idx, oob_idx in zip(self.estimators_, self.feature_indices_, self.oob_indices_):
            if len(oob_idx) == 0:
                continue

            X_oob = X_array[oob_idx].copy()
            if permuted_feature is not None:
                if rng is None:
                    raise ValueError("rng is required for feature permutation.")
                X_oob[:, permuted_feature] = rng.permutation(X_oob[:, permuted_feature])

            probabilities[oob_idx] += self._aligned_tree_proba(tree, X_oob[:, feature_idx])
            counts[oob_idx] += 1

        safe_counts = counts.copy()
        safe_counts[safe_counts == 0] = 1
        probabilities = probabilities / safe_counts[:, None]
        return probabilities, counts

    def _score_oob_predictions(self, y: np.ndarray) -> float:
        mask = self.oob_counts_ > 0
        if not mask.any():
            return float("nan")
        predictions = self.classes_[np.argmax(self.oob_decision_function_[mask], axis=1)]
        return float(accuracy_score(y[mask], predictions))

    def _aligned_tree_proba(self, tree: DecisionTreeClassifier, X_array: np.ndarray) -> np.ndarray:
        raw = tree.predict_proba(X_array)
        aligned = np.zeros((len(X_array), self.n_classes_), dtype=float)
        for tree_col, class_label in enumerate(tree.classes_):
            class_col = int(np.where(self.classes_ == class_label)[0][0])
            aligned[:, class_col] = raw[:, tree_col]
        return aligned

    def _mean_impurity_importances(self) -> np.ndarray:
        importances = np.zeros(self.n_features_in_, dtype=float)
        for tree, feature_idx in zip(self.estimators_, self.feature_indices_):
            importances[feature_idx] += tree.feature_importances_
        importances /= len(self.estimators_)
        total = importances.sum()
        return importances / total if total > 0 else importances

    def _check_fitted(self) -> None:
        if not hasattr(self, "estimators_") or not self.estimators_:
            raise RuntimeError("The forest is not fitted yet.")


def oob_accuracy_scorer(estimator: OOBRandomForestClassifier, X: Any, y: Any) -> float:
    score = estimator.oob_score_
    return 0.0 if np.isnan(score) else float(score)
