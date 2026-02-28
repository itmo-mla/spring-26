import numpy as np
from typing import Tuple


def train_test_split_bootstrap(
    X: np.ndarray, 
    y: np.ndarray, 
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    
    train_indices = rng.choice(n_samples, size=n_samples, replace=True)
    
    oob_mask = np.ones(n_samples, dtype=bool)
    oob_mask[train_indices] = False
    oob_indices = np.where(oob_mask)[0]
    
    return train_indices, oob_indices


def get_random_subspace_indices(
    n_features: int, 
    max_features: int = None,
    random_state: int = None
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    
    if max_features is None or max_features >= n_features:
        max_features = n_features
        
    max_features = max(1, min(max_features, n_features))
    
    return rng.choice(n_features, size=max_features, replace=False)  # Индексы выбранных признаков


def oob_score(
    X: np.ndarray, 
    y: np.ndarray, 
    trees: list, 
    oob_indices_per_tree: list,
    n_classes: int = None
) -> float:
    n_samples = X.shape[0]
    if n_classes is None:
        n_classes = len(np.unique(y))
    
    oob_predictions = np.zeros((n_samples, n_classes))
    oob_counts = np.zeros(n_samples)
    
    for tree, oob_indices in zip(trees, oob_indices_per_tree):
        if len(oob_indices) == 0:
            continue
            
        X_oob = X[oob_indices]
        pred_proba = tree.predict_proba(X_oob)
        
        for i, idx in enumerate(oob_indices):
            oob_predictions[idx] += pred_proba[i]
            oob_counts[idx] += 1
    
    valid_mask = oob_counts > 0  # Учитываем только объекты с хотя бы одним OOB предсказанием
    if not np.any(valid_mask):
        return 0.0
    
    pred_labels = np.argmax(oob_predictions[valid_mask], axis=1)
    true_labels = y[valid_mask]
    
    return np.mean(pred_labels == true_labels)


def permutation_importance_oob(
    X: np.ndarray, 
    y: np.ndarray,
    trees: list,
    oob_indices_per_tree: list,
    feature_indices: np.ndarray,
    n_repeats: int = 1,
    random_state: int = None
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    importances = np.zeros(n_features)
    
    base_score = oob_score(X, y, trees, oob_indices_per_tree)
    
    for j in range(n_features):
        total_drop = 0.0
        count = 0
        
        for _ in range(n_repeats):
            X_permuted = X.copy()
            perm_indices = rng.permutation(X.shape[0])
            X_permuted[:, j] = X[perm_indices, j]
            
            perm_score = _oob_score_permuted(
                X_permuted, y, trees, oob_indices_per_tree
            )
            total_drop += base_score - perm_score
            count += 1
        
        importances[j] = total_drop / count if count > 0 else 0
    
    return importances


def _oob_score_permuted(
    X: np.ndarray, 
    y: np.ndarray, 
    trees: list, 
    oob_indices_per_tree: list
) -> float:
    """
    OOB score с одним пермутированным признаком.
    """
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    
    oob_predictions = np.zeros((n_samples, n_classes))
    oob_counts = np.zeros(n_samples)
    
    for tree, oob_indices in zip(trees, oob_indices_per_tree):
        if len(oob_indices) == 0:
            continue
            
        X_oob = X[oob_indices]
        pred_proba = tree.predict_proba(X_oob)
        
        for i, idx in enumerate(oob_indices):
            oob_predictions[idx] += pred_proba[i]
            oob_counts[idx] += 1
    
    valid_mask = oob_counts > 0
    if not np.any(valid_mask):
        return 0.0
    
    pred_labels = np.argmax(oob_predictions[valid_mask], axis=1)
    true_labels = y[valid_mask]
    
    return np.mean(pred_labels == true_labels)
