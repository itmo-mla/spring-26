from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseEnsemble(BaseEstimator, ClassifierMixin, ABC):
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Any = "sqrt",
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        oob_score: bool = False,
        **kwargs
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self._max_features = None
        self._estimators = []
        self._oob_score_ = None
        self._feature_importances_ = None
        
    @abstractmethod
    def _make_estimator(self, random_state: int) -> BaseEstimator:
        pass
    
    @abstractmethod
    def _get_feature_subset(self, n_features: int, random_state: int) -> np.ndarray:
        pass
    
    def _compute_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return int(np.sqrt(n_features))
            elif self.max_features == "log2":
                return int(np.log2(n_features))
            else:
                raise ValueError(f"Unknown max_features: {self.max_features}")
        elif isinstance(self.max_features, (int, float)):
            if isinstance(self.max_features, float):
                return max(1, int(self.max_features * n_features))
            return self.max_features
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEnsemble':
        n_samples, n_features = X.shape
        self._max_features = self._compute_max_features(n_features)
        
        rng = np.random.RandomState(self.random_state)
        self._estimators = []

        oob_predictions = [] if self.oob_score else None
        oob_indices_list = [] if self.oob_score else None
        
        for i in range(self.n_estimators):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))

            feature_indices = self._get_feature_subset(n_features, rng.randint(0, 2**31))

            estimator = self._make_estimator(random_state=rng.randint(0, 2**31))
            X_bootstrap = X[indices][:, feature_indices]
            y_bootstrap = y[indices]
            
            estimator.fit(X_bootstrap, y_bootstrap)
            estimator._feature_indices = feature_indices
            estimator._oob_indices = oob_indices
            
            self._estimators.append(estimator)
            
            if self.oob_score and len(oob_indices) > 0:
                oob_predictions.append(estimator.predict(X[oob_indices][:, feature_indices]))
                oob_indices_list.append(oob_indices)

        if self.oob_score:
            self._compute_oob_score(X, y, oob_predictions, oob_indices_list)
        
        return self
    
    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray, 
                          oob_predictions: list, oob_indices_list: list) -> None:
        n_samples = X.shape[0]
        oob_votes = np.zeros(n_samples, dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int32)
        
        for pred, indices in zip(oob_predictions, oob_indices_list):
            oob_votes[indices] += pred
            oob_counts[indices] += 1

        valid_mask = oob_counts > 0
        if np.any(valid_mask):
            oob_pred = (oob_votes[valid_mask] > oob_counts[valid_mask] / 2).astype(np.int32)
            self._oob_score_ = np.mean(oob_pred == y[valid_mask])
        else:
            self._oob_score_ = 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        votes = np.zeros(n_samples, dtype=np.float64)
        
        for estimator in self._estimators:
            feature_indices = estimator._feature_indices
            pred = estimator.predict(X[:, feature_indices])
            votes += pred

        return (votes > self.n_estimators / 2).astype(np.int32)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, 2), dtype=np.float64)
        
        for estimator in self._estimators:
            feature_indices = estimator._feature_indices
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X[:, feature_indices])
                votes += proba
            else:
                pred = estimator.predict(X[:, feature_indices])
                votes[np.arange(n_samples), (pred + 1) // 2] += 1
        
        votes /= self.n_estimators
        return votes
