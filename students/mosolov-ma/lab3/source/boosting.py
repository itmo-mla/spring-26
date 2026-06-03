from dataclasses import dataclass, field

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GradientBoostingBinaryClassifier:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    subsample: float = 0.8
    random_state: int = 42
    estimators_: list[DecisionTreeRegressor] = field(default_factory=list, init=False)
    alphas_: list[float] = field(default_factory=list, init=False)
    train_scores_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingBinaryClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        y_signed = np.where(y > 0, 1.0, -1.0)

        rng = np.random.default_rng(self.random_state)
        raw_pred = np.zeros(len(X), dtype=float)

        self.estimators_.clear()
        self.alphas_.clear()
        self.train_scores_.clear()

        for _ in range(self.n_estimators):
            negative_gradient = y_signed / (1.0 + np.exp(y_signed * raw_pred))
            sample_indices = self._sample_indices(len(X), rng)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            ).fit(X[sample_indices], negative_gradient[sample_indices])

            tree_pred = tree.predict(X)
            alpha = self._line_search(raw_pred, tree_pred, y_signed, sample_indices)

            raw_pred = raw_pred + self.learning_rate * alpha * tree_pred
            self.estimators_.append(tree)
            self.alphas_.append(float(alpha))
            self.train_scores_.append(self._logistic_loss(raw_pred, y_signed))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        score = np.zeros(len(X), dtype=float)
        for alpha, tree in zip(self.alphas_, self.estimators_):
            score += self.learning_rate * alpha * tree.predict(X)
        return score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        proba_pos = _sigmoid(scores)
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)

    def _sample_indices(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n_samples)
        sample_size = max(2, int(round(self.subsample * n_samples)))
        return np.sort(rng.choice(n_samples, size=sample_size, replace=False))

    def _line_search(
        self,
        raw_pred: np.ndarray,
        tree_pred: np.ndarray,
        y_signed: np.ndarray,
        sample_indices: np.ndarray,
    ) -> float:
        if len(sample_indices) == 0:
            return 0.0

        raw_subset = raw_pred[sample_indices]
        tree_subset = tree_pred[sample_indices]
        y_subset = y_signed[sample_indices]

        def objective(alpha: float) -> float:
            margin = y_subset * (raw_subset + alpha * tree_subset)
            return float(np.mean(np.logaddexp(0.0, -margin)))

        candidates = np.linspace(0.0, 5.0, 51)
        losses = [objective(alpha) for alpha in candidates]
        return float(candidates[int(np.argmin(losses))])

    @staticmethod
    def _logistic_loss(raw_pred: np.ndarray, y_signed: np.ndarray) -> float:
        margin = y_signed * raw_pred
        return float(np.mean(np.logaddexp(0.0, -margin)))
