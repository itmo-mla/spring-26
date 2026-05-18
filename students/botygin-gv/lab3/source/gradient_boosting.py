import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import expit


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, random_state: int = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.trees_ = []
        self.init_log_odds_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        y = np.asarray(y, dtype=float)

        p = np.mean(y)
        self.init_log_odds_ = np.log(p / (1 - p + 1e-10))
        logits = np.full(len(y), self.init_log_odds_)

        self.trees_ = []

        for t in range(self.n_estimators):
            neg_gradient = y - expit(logits)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(X, neg_gradient)

            logits += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

        return self

    def _decision_function(self, X):
        logits = np.full(X.shape[0], self.init_log_odds_)
        for tree in self.trees_:
            logits += self.learning_rate * tree.predict(X)
        return logits

    def predict_proba(self, X):
        probs = expit(self._decision_function(X))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)