from __future__ import annotations

from typing import Tuple

import kagglehub
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET_PATH_STRING = "fedesoriano/california-housing-prices-data-extra-features"
DATASET_FILENAME = "California_Houses.csv"


def load_raw_dataframe() -> pd.DataFrame:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_PATH_STRING,
        DATASET_FILENAME,
    )
    return df


def load_dataset(
    test_size: float = 0.3,
    random_state: int = 42
):
    df = load_raw_dataframe()

    X = df.drop(columns=["Median_House_Value"])
    y = df["Median_House_Value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train.values, X_test.values, y_train.values, y_test.values, df
