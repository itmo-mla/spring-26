import numpy as np
from models.random_forest import RandomForest


def compute_feature_importance(model: RandomForest, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return model.compute_feature_importance_oob(X, y)
