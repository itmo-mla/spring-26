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
    description: dict[str, int | str]


def load_cancer_dataset(test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    """Load Breast Cancer Wisconsin and make malignant the positive class."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.frame.drop(columns=["target"]).copy()

    # sklearn encodes malignant as 0 and benign as 1. For reporting, use 1 = malignant.
    y = (1 - dataset.target.to_numpy(dtype=int)).astype(int)

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
        feature_names=list(X.columns),
        description={
            "name": "sklearn Breast Cancer Wisconsin",
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "classes": 2,
            "positive_class": "malignant",
        },
    )
