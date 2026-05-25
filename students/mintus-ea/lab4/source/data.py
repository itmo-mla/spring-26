from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_all: np.ndarray
    y_all: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    description: dict[str, int | str]


def load_wine_density_dataset(test_size: float = 0.25, random_state: int = 42) -> DatasetBundle:
    """Load and standardize Wine data for density estimation with GMM."""
    dataset = load_wine(as_frame=True)
    X_frame = dataset.frame.drop(columns=["target"])
    y = dataset.target.to_numpy(dtype=int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_frame,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_all = scaler.transform(X_frame)

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        X_all=X_all,
        y_all=y,
        feature_names=list(X_frame.columns),
        target_names=[str(name) for name in dataset.target_names],
        description={
            "name": "sklearn Wine recognition",
            "samples": int(X_frame.shape[0]),
            "features": int(X_frame.shape[1]),
            "classes": int(len(np.unique(y))),
            "preprocessing": "StandardScaler fitted on train split",
        },
    )


def make_projection(X_train: np.ndarray, X_test: np.ndarray, X_all: np.ndarray):
    """Return a PCA projection fitted on train data for visualization only."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    train_2d = pca.fit_transform(X_train)
    test_2d = pca.transform(X_test)
    all_2d = pca.transform(X_all)
    return pca, train_2d, test_2d, all_2d
