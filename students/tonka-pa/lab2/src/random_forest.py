from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class MyRandomForestClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        oob_score: bool = True,
        class_weight: dict[Any, float] | str | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> MyRandomForestClassifier:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if self.oob_score and not self.bootstrap:
            raise ValueError("OOB score requires bootstrap=True.")

        feature_names = self._extract_feature_names(X)
        X_checked, y_checked = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.n_features_in_ = X_checked.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        self.classes_ = unique_labels(y_checked)
        self.n_classes_ = len(self.classes_)
        self._class_index_ = {
            class_label: idx for idx, class_label in enumerate(self.classes_)
        }

        rng = np.random.default_rng(self.random_state)
        sample_size = self._resolve_sample_size(X_checked.shape[0])

        self.estimators_: list[DecisionTreeClassifier] = []
        self.estimators_samples_: list[np.ndarray] = []
        self.estimators_oob_indices_: list[np.ndarray] = []

        for _ in range(self.n_estimators):
            sample_indices, oob_indices = self._draw_sample_indices(
                rng,
                X_checked.shape[0],
                sample_size,
            )
            tree_seed = int(rng.integers(np.iinfo(np.int32).max))
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                random_state=tree_seed,
            )
            tree.fit(X_checked[sample_indices], y_checked[sample_indices])

            self.estimators_.append(tree)
            self.estimators_samples_.append(sample_indices)
            self.estimators_oob_indices_.append(oob_indices)

        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_],
            axis=0,
        )

        if self.oob_score:
            self._compute_oob_predictions(X_checked, y_checked)
        else:
            self.oob_score_ = None
            self.oob_decision_function_ = None
            self.oob_counts_ = None

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_checked = check_array(X, accept_sparse=False, dtype=np.float64)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_checked.shape[1]}."
            )

        proba_sum = np.zeros((X_checked.shape[0], self.n_classes_), dtype=float)
        for tree in self.estimators_:
            proba_sum += self._aligned_tree_proba(tree, X_checked)
        return proba_sum / len(self.estimators_)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: Any, y: Any) -> float:
        return accuracy_score(y, self.predict(X))

    def compute_oob_permutation_importance(
        self,
        X: Any,
        y: Any,
        feature_names: list[str] | np.ndarray | None = None,
        n_repeats: int = 1,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Estimate feature importance as mean OOB error increase."""
        check_is_fitted(self, "estimators_")
        if not self.bootstrap:
            raise ValueError("OOB permutation importance requires bootstrap=True.")
        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive.")

        X_checked, y_checked = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_checked.shape[1]}."
            )

        names = self._resolve_feature_names(feature_names)
        rng = np.random.default_rng(
            self.random_state if random_state is None else random_state
        )
        drops: list[list[float]] = [[] for _ in range(self.n_features_in_)]
        relatives: list[list[float]] = [[] for _ in range(self.n_features_in_)]
        used_oob_sets = 0

        for tree, oob_indices in zip(
            self.estimators_,
            self.estimators_oob_indices_,
            strict=True,
        ):
            if len(oob_indices) == 0:
                continue
            X_oob = X_checked[oob_indices]
            y_oob = y_checked[oob_indices]
            baseline_error = 1.0 - accuracy_score(y_oob, tree.predict(X_oob))
            used_oob_sets += 1

            for feature_idx in range(self.n_features_in_):
                for _ in range(n_repeats):
                    X_permuted = X_oob.copy()
                    X_permuted[:, feature_idx] = rng.permutation(
                        X_permuted[:, feature_idx]
                    )
                    permuted_error = 1.0 - accuracy_score(
                        y_oob,
                        tree.predict(X_permuted),
                    )
                    increase = permuted_error - baseline_error
                    drops[feature_idx].append(increase)
                    if baseline_error > 0:
                        relatives[feature_idx].append(increase / baseline_error * 100)

        rows = []
        for feature_idx, values in enumerate(drops):
            relative_values = relatives[feature_idx]
            rows.append(
                {
                    "feature": names[feature_idx],
                    "importance": float(np.mean(values)) if values else 0.0,
                    "std": float(np.std(values)) if values else 0.0,
                    "relative_importance_percent": (
                        float(np.mean(relative_values)) if relative_values else np.nan
                    ),
                    "n_oob_evaluations": len(values),
                    "n_oob_tree_sets": used_oob_sets,
                }
            )

        result = pd.DataFrame(rows).sort_values("importance", ascending=False)
        self.oob_permutation_importances_ = result.reset_index(drop=True)
        return self.oob_permutation_importances_

    def _compute_oob_predictions(self, X: np.ndarray, y: np.ndarray) -> None:
        proba_sum = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        counts = np.zeros(X.shape[0], dtype=int)

        for tree, oob_indices in zip(
            self.estimators_,
            self.estimators_oob_indices_,
            strict=True,
        ):
            if len(oob_indices) == 0:
                continue
            proba_sum[oob_indices] += self._aligned_tree_proba(tree, X[oob_indices])
            counts[oob_indices] += 1

        decision = np.zeros_like(proba_sum)
        available = counts > 0
        decision[available] = proba_sum[available] / counts[available, None]

        if not np.all(available):
            missing = int((~available).sum())
            warnings.warn(
                f"{missing} samples have no OOB prediction and were excluded.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.oob_decision_function_ = decision
        self.oob_counts_ = counts
        self.oob_score_ = (
            accuracy_score(
                y[available], self.classes_[np.argmax(decision[available], axis=1)]
            )
            if np.any(available)
            else np.nan
        )

    def _aligned_tree_proba(
        self,
        tree: DecisionTreeClassifier,
        X: np.ndarray,
    ) -> np.ndarray:
        tree_proba = tree.predict_proba(X)
        aligned = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for tree_col, class_label in enumerate(tree.classes_):
            aligned[:, self._class_index_[class_label]] = tree_proba[:, tree_col]
        return aligned

    def _draw_sample_indices(
        self,
        rng: np.random.Generator,
        n_samples: int,
        sample_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.bootstrap:
            indices = np.arange(n_samples)
            return indices, np.array([], dtype=int)

        sample_indices = rng.integers(0, n_samples, size=sample_size)
        in_bag = np.zeros(n_samples, dtype=bool)
        in_bag[sample_indices] = True
        oob_indices = np.flatnonzero(~in_bag)
        return sample_indices, oob_indices

    def _resolve_sample_size(self, n_samples: int) -> int:
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, int):
            if not 1 <= self.max_samples <= n_samples:
                raise ValueError("Integer max_samples must be in [1, n_samples].")
            return self.max_samples
        if isinstance(self.max_samples, float):
            if not 0 < self.max_samples <= 1:
                raise ValueError("Float max_samples must be in (0, 1].")
            return max(1, int(round(self.max_samples * n_samples)))
        raise TypeError("max_samples must be None, int, or float.")

    def _resolve_feature_names(
        self,
        feature_names: list[str] | np.ndarray | None,
    ) -> list[str]:
        if feature_names is not None:
            names = list(feature_names)
        elif hasattr(self, "feature_names_in_"):
            names = list(self.feature_names_in_)
        else:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        if len(names) != self.n_features_in_:
            raise ValueError("feature_names length must match n_features_in_.")
        return names

    @staticmethod
    def _extract_feature_names(X: Any) -> np.ndarray | None:
        columns = getattr(X, "columns", None)
        if columns is None:
            return None
        return np.asarray(columns, dtype=object)
