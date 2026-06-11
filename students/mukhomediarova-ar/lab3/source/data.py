from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    source_name: str


def make_splits(test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    """Load Breast Cancer Wisconsin and return a reproducible stratified split."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.frame[dataset.feature_names].copy()
    y = dataset.target.to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetBundle(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        feature_names=list(dataset.feature_names),
        target_names=list(dataset.target_names),
        source_name="Breast Cancer Wisconsin (Diagnostic), sklearn.datasets.load_breast_cancer",
    )
