from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


class StochasticGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 0.7,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Поддерживается только бинарная классификация.")

        y_pos = (y == self.classes_[1]).astype(np.float64, copy=False)
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample должен быть в (0, 1].")

        p_bar = float(np.clip(y_pos.mean(), 1e-6, 1.0 - 1e-6))
        self.init_prediction_ = np.log(p_bar / (1.0 - p_bar))
        F = np.full(n_samples, self.init_prediction_, dtype=np.float64)

        self.estimators_: list[DecisionTreeRegressor] = []
        n_sub = max(1, int(self.subsample * n_samples))

        for m in range(self.n_estimators):
            p = _sigmoid(F)
            residuals = y_pos - p
            idx = rng.choice(n_samples, size=n_sub, replace=False)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.randint(0, 2**31 - 1)),
            )
            tree.fit(X[idx], residuals[idx])
            F = F + self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)

        self.n_features_in_ = X.shape[1]
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        scores = np.full(X.shape[0], self.init_prediction_, dtype=np.float64)
        for tree in self.estimators_:
            scores += self.learning_rate * tree.predict(X)
        return scores

    def predict_proba(self, X):
        p1 = _sigmoid(self._decision_function(X))
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        pos = self.classes_[1]
        neg = self.classes_[0]
        return np.where(proba >= 0.5, pos, neg)
