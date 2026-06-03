import numpy as np
from src.stump import DecisionStump


def _softmax(F: np.ndarray) -> np.ndarray:
    e = np.exp(F - F.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class GradientBoostingClassifier:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        n = X.shape[0]
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        F = np.zeros((n, K))

        self.estimators_: list[list[DecisionStump]] = []
        self.train_scores_: list[float] = []

        for _ in range(self.n_estimators):
            p = _softmax(F)
            residuals = Y - p

            stumps = []
            for k in range(K):
                s = DecisionStump().fit(X, residuals[:, k])
                F[:, k] += self.learning_rate * s.predict(X)
                stumps.append(s)

            self.estimators_.append(stumps)
            acc = (F.argmax(axis=1) == y).mean()
            self.train_scores_.append(float(acc))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        K = len(self.classes_)
        F = np.zeros((X.shape[0], K))
        for stumps in self.estimators_:
            for k, s in enumerate(stumps):
                F[:, k] += self.learning_rate * s.predict(X)
        return _softmax(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
