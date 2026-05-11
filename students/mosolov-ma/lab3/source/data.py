from dataclasses import dataclass

import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


DATASET_NAME = "nareshbhat/health-care-data-set-on-heart-attack-possibility"
DATASET_FILE = "heart.csv"


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str


def load_heart_dataset() -> pd.DataFrame:
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_NAME,
        DATASET_FILE,
    )


def prepare_dataset(df: pd.DataFrame, target_col: str = "target") -> DatasetBundle:
    if target_col not in df.columns:
        target_col = df.columns[-1]

    y = df[target_col].astype(int).to_numpy()
    X_df = df.drop(columns=[target_col]).copy()
    feature_names = list(X_df.columns)
    X = X_df.astype(float).to_numpy()

    return DatasetBundle(
        dataframe=df,
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_col,
    )
