from sklearn.base import BaseEstimator, ClassifierMixin
from source.RandomForest import RandomForest

class SklearnRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, max_depth=None, max_features='sqrt', min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.rf = None

    def fit(self, X, y):
        self.rf = RandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self.rf.fit(X, y)
        return self

    def predict(self, X):
        return self.rf.predict(X)

    def score(self, X, y=None):
        return self.rf.oob_score(X, y)