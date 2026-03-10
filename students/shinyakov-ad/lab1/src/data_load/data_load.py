from __future__ import annotations

from typing import Tuple

import kagglehub
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET_PATH_STRING = "fedesoriano/stroke-prediction-dataset"
DATASET_FILENAME = "healthcare-dataset-stroke-data.csv"


def load_raw_dataframe() -> pd.DataFrame:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_PATH_STRING,
        DATASET_FILENAME,
    )
    return df


def _encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    encoders = {}

    df_encoded = df.copy()
    for col in cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    return df_encoded


def load_dataset(
    test_size: float = 0.3,
    random_state: int = 42
):
    df = load_raw_dataframe()
    df = df.drop(columns=["id"])
    df_encoded = _encode_categorical(df)

    X = df_encoded.drop(columns=["stroke"])
    y = df_encoded["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train.values, X_test.values, y_train.values, y_test.values, df_encoded
