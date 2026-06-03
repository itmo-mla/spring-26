from typing import Tuple

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def load_dataset(
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:

    data = load_wine()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)

    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, y, feature_names, target_names
