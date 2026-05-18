from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeRegressor

from utils.training import fit_grid_search


class GradientBoostingClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        gamma_grid_size: int = 3,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.gamma_grid_size = gamma_grid_size

    @staticmethod
    def _sigmoid(raw: np.ndarray) -> np.ndarray:
        raw = np.clip(raw, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-raw))

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.classes_ = np.sort(y.unique())
        if len(self.classes_) != 2:
            raise ValueError("Поддерживается только бинарная классификация")

        self.feature_names_in_ = X.columns.to_numpy()
        self.n_features_in_ = X.shape[1]

        y_bin = (y == self.classes_[1]).astype(int).to_numpy()
        positive_rate = np.clip(y_bin.mean(), 1e-8, 1 - 1e-8)
        self.init_prediction_ = float(np.log(positive_rate / (1.0 - positive_rate)))

        self.estimators_: List[DecisionTreeRegressor] = []
        self.gammas_: List[float] = []
        self.train_loss_: List[float] = []

        raw_prediction = np.full(len(X), self.init_prediction_, dtype=float)
        rng = np.random.RandomState(self.random_state)
        gamma_candidates = np.linspace(0.0, 1.0, self.gamma_grid_size)

        for _ in range(self.n_estimators):
            probability = self._sigmoid(raw_prediction)
            residual = y_bin - probability

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.randint(0, 10**9)),
            )
            tree.fit(X, residual)
            update = tree.predict(X)

            best_gamma = 0.0
            best_loss = np.inf
            for gamma in gamma_candidates:
                candidate_raw_prediction = raw_prediction + self.learning_rate * gamma * update
                candidate_probability = self._sigmoid(candidate_raw_prediction)
                loss = log_loss(y_bin, candidate_probability, labels=[0, 1])
                if loss < best_loss:
                    best_loss = loss
                    best_gamma = float(gamma)

            raw_prediction = raw_prediction + self.learning_rate * best_gamma * update
            self.estimators_.append(tree)
            self.gammas_.append(best_gamma)
            self.train_loss_.append(best_loss)

        self.n_estimators_ = len(self.estimators_)
        return self

    def _raw_predict(self, X) -> np.ndarray:
        X = pd.DataFrame(X).reset_index(drop=True)
        raw_prediction = np.full(len(X), self.init_prediction_, dtype=float)
        for tree, gamma in zip(self.estimators_, self.gammas_):
            raw_prediction += self.learning_rate * gamma * tree.predict(X)
        return raw_prediction

    def predict_proba(self, X):
        raw_prediction = self._raw_predict(X)
        positive_probability = self._sigmoid(raw_prediction)
        negative_probability = 1.0 - positive_probability
        return np.column_stack([negative_probability, positive_probability])

    def predict(self, X):
        positive_probability = self.predict_proba(X)[:, 1]
        indices = (positive_probability >= 0.5).astype(int)
        return self.classes_[indices]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))



def get_my_boosting(X_train: pd.DataFrame, y_train: pd.Series):
    estimator = GradientBoostingClassifierCustom(random_state=42)
    return fit_grid_search(estimator, X_train, y_train)
