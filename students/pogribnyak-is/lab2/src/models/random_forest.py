from typing import Optional, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from models.base import BaseEnsemble


class RandomForest(BaseEnsemble):
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Any = "sqrt",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        oob_score: bool = False,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=oob_score
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def _make_estimator(self, random_state: int) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state
        )
    
    def _transform_labels_for_tree(self, y: np.ndarray) -> np.ndarray:
        return (y + 1) // 2
    
    def _transform_predictions_from_tree(self, pred: np.ndarray) -> np.ndarray:
        return pred * 2 - 1
    
    def _get_feature_subset(self, n_features: int, random_state: int) -> np.ndarray:
        rng = np.random.RandomState(random_state)
        n_selected = self._max_features
        return rng.choice(n_features, size=n_selected, replace=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        y_transformed = self._transform_labels_for_tree(y)
        return super().fit(X, y_transformed)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = super().predict(X)
        return self._transform_predictions_from_tree(pred)
    
    def compute_feature_importance_oob(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape

        y_transformed = self._transform_labels_for_tree(y)

        base_oob_score = self._oob_score_ if self._oob_score_ is not None else 0.0
        
        feature_importances = np.zeros(n_features)
        
        for j in range(n_features):
            X_permuted = X.copy()
            seed = (self.random_state if self.random_state is not None else 42) + j
            rng = np.random.RandomState(seed)
            perm_indices = rng.permutation(n_samples)
            X_permuted[:, j] = X[perm_indices, j]

            oob_predictions = []
            oob_indices_list = []
            
            for estimator in self._estimators:
                oob_indices = estimator._oob_indices
                if len(oob_indices) > 0:
                    feature_indices = estimator._feature_indices
                    if j in feature_indices:
                        X_oob = X_permuted[oob_indices][:, feature_indices]
                        pred = estimator.predict(X_oob)
                        oob_predictions.append(pred)
                        oob_indices_list.append(oob_indices)
            
            if oob_predictions:
                oob_votes = np.zeros(n_samples, dtype=np.float64)
                oob_counts = np.zeros(n_samples, dtype=np.int32)
                
                for pred, indices in zip(oob_predictions, oob_indices_list):
                    oob_votes[indices] += pred
                    oob_counts[indices] += 1
                
                valid_mask = oob_counts > 0
                if np.any(valid_mask):
                    oob_pred = (oob_votes[valid_mask] > oob_counts[valid_mask] / 2).astype(np.int32)
                    permuted_oob_score = np.mean(oob_pred == y_transformed[valid_mask])
                    feature_importances[j] = base_oob_score - permuted_oob_score

        total = np.sum(feature_importances)
        if total > 0:
            feature_importances /= total
        
        self._feature_importances_ = feature_importances
        return feature_importances
