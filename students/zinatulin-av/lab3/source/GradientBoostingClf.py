import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingClf(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    @staticmethod
    def _loss(a, y):
        return np.logaddexp(0.0, -a * y)

    @staticmethod
    def _grad(a, y):
        return -y / (1.0 + np.exp(a * y))

    def fit(self, X, y):
        y = np.where(np.asarray(y) > 0, 1.0, -1.0)
        self.trees_ = []
        self.alphas_ = []
        a = np.zeros(len(y))

        for _ in range(self.n_estimators):
            anti_grad = -self._grad(a, y)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, anti_grad)
            b = tree.predict(X)

            res = minimize_scalar(
                lambda alpha: self._loss(a + alpha * b, y).sum(),
                bounds=(0.0, 10.0),
                method='bounded',
            )
            alpha = res.x

            a = a + alpha * b
            self.trees_.append(tree)
            self.alphas_.append(alpha)

        return self

    def decision_function(self, X):
        return sum(alpha * tree.predict(X) for alpha, tree in zip(self.alphas_, self.trees_))

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def predict_proba(self, X):
        f = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-2.0 * f))
        return np.column_stack([1.0 - p, p])
