import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor


class GradientBoosting(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees_ = []
        self.init_prediction_ = None
        self.train_losses_ = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _log_loss(y: np.ndarray, proba: np.ndarray) -> float:
        eps = 1e-15
        proba = np.clip(proba, eps, 1 - eps)
        return -np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        y = y.astype(float)
        self.classes_ = np.array([0, 1], dtype=int)
        rng = np.random.default_rng(self.random_state)

        p = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        self.init_prediction_ = np.log(p / (1 - p))

        F = np.full(n_samples, self.init_prediction_)
        self.trees_ = []
        self.train_losses_ = []

        for _ in range(self.n_estimators):
            proba = self._sigmoid(F)
            residuals = y - proba

            if self.subsample < 1.0:
                sample_size = max(1, int(n_samples * self.subsample))
                idx = rng.choice(n_samples, size=sample_size, replace=False)
            else:
                idx = np.arange(n_samples)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[idx], residuals[idx])

            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees_.append(tree)

            loss = self._log_loss(y, self._sigmoid(F))
            self.train_losses_.append(loss)

        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        F = np.full(X.shape[0], self.init_prediction_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba_pos = self._sigmoid(self._raw_predict(X))
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._sigmoid(self._raw_predict(X)) >= 0.5).astype(int)
