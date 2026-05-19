from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    description: dict[str, int | str]


def load_digits_dataset(test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    """Load the built-in 8x8 handwritten digits dataset."""
    digits = load_digits(as_frame=True)
    X = digits.frame.drop(columns=["target"]).copy()
    y = digits.target.to_numpy(dtype=int)

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
        target_names=[str(name) for name in digits.target_names],
        description={
            "name": "sklearn Optical Recognition of Handwritten Digits",
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "classes": int(len(np.unique(y))),
            "image_shape": "8x8",
        },
    )
