from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, float, int] = "sqrt",
        bootstrap: bool = True,
        criterion: str = "gini",
        random_state: int = 42,
        compute_oob_importance: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.random_state = random_state
        self.compute_oob_importance = compute_oob_importance

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        self.trees_: List[DecisionTreeClassifier] = []
        self.oob_indices_: List[np.ndarray] = []

        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            seed = int(rng.randint(0, 10**9))
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )

            if self.bootstrap:
                sample_idx = rng.choice(len(X), size=len(X), replace=True)
                used = np.zeros(len(X), dtype=bool)
                used[sample_idx] = True
                oob_idx = np.where(~used)[0]
            else:
                sample_idx = np.arange(len(X))
                oob_idx = np.array([], dtype=int)

            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
            self.oob_indices_.append(oob_idx)

        self.oob_score_ = self._compute_oob_score(X, y)
        if self.compute_oob_importance:
            self.feature_importances_oob_ = self._compute_oob_feature_importance(X, y)
        else:
            self.feature_importances_oob_ = {}

        return self

    def compute_feature_importances_oob(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.feature_importances_oob_ = self._compute_oob_feature_importance(X, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        proba_sum = np.zeros((len(X), self.n_classes_), dtype=float)

        for tree in self.trees_:
            tree_proba = tree.predict_proba(X)
            aligned = np.zeros((len(X), self.n_classes_), dtype=float)
            for idx, label in enumerate(tree.classes_):
                class_idx = int(np.where(self.classes_ == label)[0][0])
                aligned[:, class_idx] = tree_proba[:, idx]
            proba_sum += aligned

        return proba_sum / len(self.trees_)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _align_proba(self, tree: DecisionTreeClassifier, X_part: np.ndarray) -> np.ndarray:
        tree_proba = tree.predict_proba(X_part)
        aligned = np.zeros((len(X_part), self.n_classes_), dtype=float)
        for idx, label in enumerate(tree.classes_):
            class_idx = int(np.where(self.classes_ == label)[0][0])
            aligned[:, class_idx] = tree_proba[:, idx]
        return aligned

    def _collect_oob_predictions(
        self,
        X: np.ndarray,
        permuted_feature: Optional[int] = None,
        permuted_values: Optional[np.ndarray] = None,
    ):
        votes = np.zeros((len(X), self.n_classes_), dtype=float)
        counts = np.zeros(len(X), dtype=int)

        for tree, oob_idx in zip(self.trees_, self.oob_indices_):
            if len(oob_idx) == 0:
                continue

            X_oob = X[oob_idx].copy()
            if permuted_feature is not None:
                X_oob[:, permuted_feature] = permuted_values[oob_idx]

            aligned = self._align_proba(tree, X_oob)
            votes[oob_idx] += aligned
            counts[oob_idx] += 1

        mask = counts > 0
        counts_safe = counts.copy()
        counts_safe[counts_safe == 0] = 1
        probs = votes / counts_safe[:, None]
        pred = self.classes_[np.argmax(probs, axis=1)]
        return pred, mask

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        pred, mask = self._collect_oob_predictions(X)
        if not mask.any():
            return float("nan")
        return float(accuracy_score(y[mask], pred[mask]))

    def _compute_oob_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        baseline = self.oob_score_
        rng = np.random.RandomState(self.random_state)
        result: Dict[int, float] = {}

        for feature_idx in range(X.shape[1]):
            permuted_values = X[:, feature_idx].copy()
            rng.shuffle(permuted_values)
            pred, mask = self._collect_oob_predictions(
                X,
                permuted_feature=feature_idx,
                permuted_values=permuted_values,
            )
            if not mask.any():
                result[feature_idx] = 0.0
                continue
            permuted_score = accuracy_score(y[mask], pred[mask])
            result[feature_idx] = baseline - permuted_score

        return result


def oob_scorer(estimator, X, y):
    return estimator.oob_score_


def grid_search_forest(
    estimator: BaseEstimator,
    X,
    y,
    param_grid: dict,
    random_state: int = 42,
) -> GridSearchCV:
    train_idx = np.arange(len(X))
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=oob_scorer,
        cv=[(train_idx, train_idx)],
        refit=True,
        n_jobs=1,
    ).fit(X, y)
