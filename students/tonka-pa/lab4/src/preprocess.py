"""Data loading and preprocessing for the Santander dataset.

The raw CSV files live under ``data/raw/``. ``test.csv`` from Kaggle contains
no labels, so all train/val/test splits are derived from ``train.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


@dataclass
class DataSplits:
    """Container for a single train/val/test split with optional transforms."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler | None
    pca: PCA | None

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def load_raw(n_samples: int | None = None, random_state: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Read Santander train.csv and return the feature DataFrame and target.

    Drops ``ID_code`` and verifies that there are no missing values.
    """
    path = RAW_DIR / "train.csv"
    df = pd.read_csv(path)
    if "ID_code" in df.columns:
        df = df.drop(columns=["ID_code"])
    y = df["target"]
    X = df.drop(columns=["target"])
    if X.isna().any().any():
        raise ValueError("unexpected missing values in Santander features")
    if n_samples is not None and n_samples < len(df):
        sample = df.sample(n=n_samples, random_state=random_state)
        y = sample["target"]
        X = sample.drop(columns=["target"])
    return X.reset_index(drop=True), y.reset_index(drop=True)


def make_splits(
    *,
    n_samples: int | None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    scale: bool = True,
    pca_components: int | None = None,
    stratify: bool = True,
    random_state: int = 0,
) -> DataSplits:
    """Create a deterministic train/val/test split, optionally scaled and PCA'd."""
    X_df, y_ser = load_raw(n_samples=n_samples, random_state=random_state)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float64)
    y = y_ser.to_numpy()
    stratify_arr = y if stratify else None
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arr
    )
    stratify_tv = y_tv if stratify else None
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_fraction, random_state=random_state, stratify=stratify_tv
    )

    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    pca: PCA | None = None
    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=random_state).fit(X_train)
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        feature_names = [f"pc_{i}" for i in range(pca_components)]

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
        pca=pca,
    )
