import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin


class GB(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, criterion='friedman_mse',
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.models = []
        self.gammas = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def logloss(self, y, F):
        p = self.sigmoid(F)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def find_gamma(self, y, Fm, pred):
        gammas = np.linspace(0, 1, 100)
        losses = [
            self.logloss(y, Fm + g * pred)
            for g in gammas
        ]
        return gammas[np.argmin(losses)]

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        eps = 1e-15
        p0 = np.clip(np.mean(y), eps, 1 - eps)

        # F0 = logit(mean(y))
        self.F0 = np.log(p0 / (1 - p0))

        Fm = np.ones(len(y)) * self.F0

        self.models = []
        self.gammas = []

        for _ in range(self.n_estimators):

            p = self.sigmoid(Fm)
            residuals = y - p

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.rng.randint(0, 2**31),
            )
            tree.fit(X, residuals)

            pred = tree.predict(X)

            gamma = self.find_gamma(y, Fm, pred)

            Fm += self.learning_rate * gamma * pred

            self.models.append(tree)
            self.gammas.append(gamma)

        return self

    def decision_function(self, X):
        X = np.array(X)

        Fm = np.ones(X.shape[0]) * self.F0

        for tree, gamma in zip(self.models, self.gammas):
            Fm += self.learning_rate * gamma * tree.predict(X)

        return Fm

    def predict_proba(self, X):
        Fm = self.decision_function(X)
        p = self.sigmoid(Fm)

        return np.vstack([1 - p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)