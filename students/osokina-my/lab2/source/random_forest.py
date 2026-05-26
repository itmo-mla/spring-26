import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, List

from .utils import (
    train_test_split_bootstrap,
    get_random_subspace_indices,
    oob_score,
    permutation_importance_oob,
)


class CustomRandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Optional[str] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators  # Количество деревьев
        self.max_features = max_features  # Макс. признаков при разбиении ('sqrt', 'log2', int, float, None)
        self.max_depth = max_depth  # Максимальная глубина деревьев
        self.min_samples_split = min_samples_split  # Минимум сэмплов для разбиения
        self.min_samples_leaf = min_samples_leaf  # Минимум сэмплов в листе
        self.random_state = random_state  # Seed для воспроизводимости
        
        self.trees_: List[DecisionTreeClassifier] = []
        self.feature_indices_per_tree_: List[np.ndarray] = []
        self.oob_indices_per_tree_: List[np.ndarray] = []
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None
        self.feature_importances_: Optional[np.ndarray] = None
        
    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)) + 1)
        return n_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomRandomForestClassifier":
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features_subspace = self._resolve_max_features(self.n_features_)
        
        self.trees_ = []
        self.feature_indices_per_tree_ = []
        self.oob_indices_per_tree_ = []
        
        rng = np.random.default_rng(self.random_state)
        
        for _ in range(self.n_estimators):
            rs = rng.integers(0, 2**31 - 1) if self.random_state is not None else None

            # Bootstrap агрегирование (Bagging) и индексы, которые не вошли
            train_idx, oob_idx = train_test_split_bootstrap(X, y, random_state=rs)
            feat_idx = get_random_subspace_indices(  # Метод случайных подпространств (RSM)
                self.n_features_, 
                n_features_subspace, 
                random_state=rs
            )
            
            X_train = X[np.ix_(train_idx, feat_idx)]
            y_train = y[train_idx]
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rs,
            )
            tree.fit(X_train, y_train)
            
            self.trees_.append(tree)
            self.feature_indices_per_tree_.append(feat_idx)
            self.oob_indices_per_tree_.append(oob_idx)
        
        self.oob_score_ = oob_score(
            X, y, 
            self._get_trees_with_mapped_features(),
            self.oob_indices_per_tree_,
            n_classes
        )
        
        return self
    
    def _get_trees_with_mapped_features(self) -> list:
        """
        Создаёт обёртки деревьев, которые принимают полную матрицу X
        и передают в дерево только нужные признаки.
        """
        class TreeWrapper:
            def __init__(self, tree, feat_idx):
                self.tree = tree
                self.feat_idx = feat_idx
            def predict_proba(self, X):
                return self.tree.predict_proba(X[:, self.feat_idx])
        
        return [
            TreeWrapper(t, fi) 
            for t, fi in zip(self.trees_, self.feature_indices_per_tree_)
        ]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Простое голосование."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Усреднённые вероятности по деревьям."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n_samples, n_classes))
        
        for tree, feat_idx in zip(self.trees_, self.feature_indices_per_tree_):
            proba_sum += tree.predict_proba(X[:, feat_idx])
        
        return proba_sum / len(self.trees_)
    
    def get_feature_importances_oob(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        n_repeats: int = 1,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Важность признаков через OOB пермутацию (OOB^j).
        """
        wrapped = self._get_trees_with_mapped_features()
        
        importances = permutation_importance_oob(
            X, y, wrapped, self.oob_indices_per_tree_,
            feature_indices=np.arange(self.n_features_),
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        self.feature_importances_ = importances
        return importances


__all__ = ["CustomRandomForestClassifier"]
