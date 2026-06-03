import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin


class GradientBoosting(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.01,
        max_depth=5,
        min_samples_leaf=5,
        subsample=1.0,
        early_stopping_rounds=None,
        tol=1e-4,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol
        self.random_state = random_state
        self.trees_ = []
        self.init_prediction_ = None
        self.train_losses_ = []

    @staticmethod
    def _sigmoid(x):
        np.clip(x, -1000, 1000, out=x)
        return 1.0 / (1.0 + np.exp(-x, out=np.negative(x, out=np.empty_like(x))))

    @staticmethod
    def _log_loss(y, proba):
        eps = 1e-12
        proba = np.clip(proba, eps, 1 - eps, out=proba)
        return -np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba), out=np.empty_like(proba))

    def _compute_loss(self, y, F):
        proba = 1.0 / (1.0 + np.exp(-np.clip(F, -1000, 1000)))
        eps = 1e-12
        proba = np.clip(proba, eps, 1 - eps)
        return -np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64, order='C')
        y = np.asarray(y, dtype=np.float64)
        n_samples, _ = X.shape
        rng = np.random.default_rng(self.random_state)

        p = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
        self.init_prediction_ = np.log(p / (1 - p))

        F = np.full(n_samples, self.init_prediction_, dtype=np.float64)
        proba = np.empty(n_samples, dtype=np.float64)
        residuals = np.empty(n_samples, dtype=np.float64)
        
        self.trees_ = []
        self.train_losses_ = []
        
        use_subsample = self.subsample < 1.0
        if use_subsample:
            sample_size = max(1, int(n_samples * self.subsample))
        else:
            sample_size = n_samples

        best_loss = np.inf
        no_improve_count = 0

        for i in range(self.n_estimators):
            proba = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
            residuals = y - proba

            if use_subsample:
                idx = rng.choice(n_samples, size=sample_size, replace=False)
                X_sample = X[idx]
                y_sample = residuals[idx]
            else:
                X_sample = X
                y_sample = residuals

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

            F += self.learning_rate * tree.predict(X)

            loss = self._compute_loss(y, F)
            self.train_losses_.append(loss)

            if self.early_stopping_rounds is not None:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.early_stopping_rounds:
                        break

        return self

    def _raw_predict(self, X):
        X = np.asarray(X, dtype=np.float64, order='C')
        F = np.full(X.shape[0], self.init_prediction_, dtype=np.float64)
        
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        
        return F

    def predict_proba(self, X):
        F = self._raw_predict(X)
        proba_pos = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X):
        F = self._raw_predict(X)
        proba = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
        return (proba >= 0.5).astype(np.int32)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
