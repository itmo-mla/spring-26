from __future__ import annotations
import numpy as np
from tree import ID3Tree


class BaseEnsemble:

    def __init__(self, n_estimators: int):
        self.n_estimators = n_estimators
        self.models = []
        self.bootstrap_idx = []
        self.oob_idx = []

    def _aggregate(self, preds: np.ndarray) -> np.ndarray:
        return np.mean(preds, axis=0)

    def _oob_predict(self, X: np.ndarray):
        n_samples = X.shape[0]
        preds = np.zeros(n_samples, dtype=float)
        counts = np.zeros(n_samples, dtype=int)

        for model, oob in zip(self.models, self.oob_idx):
            if len(oob) == 0:
                continue
            pred = model.predict(X[oob])
            preds[oob] += pred
            counts[oob] += 1

        mask = counts > 0
        preds[mask] /= counts[mask]
        return preds, mask

    def oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds, mask = self._oob_predict(X)
        return float(np.mean(preds[mask] == y[mask]))


class RandomForest(BaseEnsemble):

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples: int = 5,
        max_features: str | int = "sqrt",
        random_state: int = 42,
    ):
        super().__init__(n_estimators)
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, _ = X.shape
        
        self.models = []
        self.bootstrap_idx = []
        self.oob_idx = []

        for _ in range(self.n_estimators):
            idx = self.rng.integers(0, n_samples, n_samples)
            oob = np.setdiff1d(np.arange(n_samples), idx)
            
            X_boot = X[idx]
            y_boot = y[idx]
            
            tree = ID3Tree(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                max_features=self.max_features,
                rng=self.rng
            )
            
            tree.fit(X_boot, y_boot)
            
            self.models.append(tree)
            self.bootstrap_idx.append(idx)
            self.oob_idx.append(oob)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([model.predict(X) for model in self.models])
        return np.round(self._aggregate(preds)).astype(int)

    def _oob_predict(self, X: np.ndarray):
        n_samples = X.shape[0]
        preds = np.zeros(n_samples, dtype=float)
        counts = np.zeros(n_samples, dtype=int)

        for model, oob in zip(self.models, self.oob_idx):
            if len(oob) == 0:
                continue
            pred = model.predict(X[oob])
            preds[oob] += pred
            counts[oob] += 1

        mask = counts > 0
        preds[mask] /= counts[mask]
        return preds, mask

    def oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds, mask = self._oob_predict(X)
        return float(np.mean(preds[mask] == y[mask]))

    def feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        base_score = self.oob_score(X, y)
        
        n_features = X.shape[1]
        importances = np.zeros(n_features, dtype=float)
        
        for j in range(n_features):
            X_perm = X.copy()
            
            # Get all OOB indices
            all_oob = np.concatenate([self.oob_idx[i] for i in range(len(self.models)) if len(self.oob_idx[i]) > 0])
            all_oob = np.unique(all_oob)
            
            if len(all_oob) > 0:
                perm = self.rng.permutation(all_oob)
                X_perm[all_oob, j] = X[perm, j]
            
            perm_score = self.oob_score(X_perm, y)
            importances[j] = (base_score - perm_score) / base_score
        
        return importances
    