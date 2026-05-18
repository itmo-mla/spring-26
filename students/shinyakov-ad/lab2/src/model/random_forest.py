from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


@dataclass
class TreeState:
    tree: DecisionTreeRegressor
    oob_indices: np.ndarray


class RandomForestRegressorCustom(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int | None = None,
        max_features: int | float | str | None = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.trees_: list[TreeState] = []

        for i in range(self.n_estimators):
            bootstrap_indices, oob_indices = self._sample_rows(X.shape[0])

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + i,
            )
            
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])

            self.trees_.append(
                TreeState(
                    tree=tree,
                    oob_indices=oob_indices,
                )
            )

        self.oob_prediction_ = self._compute_oob_predictions(X)
        mask = ~np.isnan(self.oob_prediction_)
        self.oob_score_ = r2_score(y[mask], self.oob_prediction_[mask]) if np.any(mask) else np.nan

        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = np.column_stack(
            [
                state.tree.predict(X)
                for state in self.trees_
            ]
        )
        return predictions.mean(axis=1)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def compute_oob_feature_importance(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        baseline = self._oob_score_for_dataset(X, y)

        importances = np.zeros(X.shape[1], dtype=float)
        for feature_idx in range(X.shape[1]):
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = self._rng.permutation(X_permuted[:, feature_idx])
            permuted_score = self._oob_score_for_dataset(X_permuted, y)
            importances[feature_idx] = baseline - permuted_score

        return importances

    def _sample_rows(self, n_rows: int):
        bootstrap_indices = self._rng.integers(0, n_rows, size=n_rows)
        in_bag_mask = np.zeros(n_rows, dtype=bool)
        in_bag_mask[bootstrap_indices] = True
        oob_indices = np.flatnonzero(~in_bag_mask)
        return bootstrap_indices, oob_indices

    def _compute_oob_predictions(self, X):
        oob_sum = np.zeros(X.shape[0], dtype=float)
        oob_count = np.zeros(X.shape[0], dtype=int)

        for state in self.trees_:
            if state.oob_indices.size == 0:
                continue
            preds = state.tree.predict(X[state.oob_indices])
            oob_sum[state.oob_indices] += preds
            oob_count[state.oob_indices] += 1

        oob_prediction = np.full(X.shape[0], np.nan, dtype=float)
        valid = oob_count > 0
        oob_prediction[valid] = oob_sum[valid] / oob_count[valid]
        return oob_prediction

    def _oob_score_for_dataset(self, X, y):
        predictions = self._compute_oob_predictions(X)
        mask = ~np.isnan(predictions)
        if not np.any(mask):
            return np.nan
        return r2_score(y[mask], predictions[mask])
