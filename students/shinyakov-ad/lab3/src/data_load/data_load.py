from __future__ import annotations

from pathlib import Path
from typing import Tuple

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET_PATH_STRING = "fedesoriano/stroke-prediction-dataset"
DATASET_FILENAME = "healthcare-dataset-stroke-data.csv"
LOCAL_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / DATASET_FILENAME


def load_raw_dataframe() -> pd.DataFrame:
    if LOCAL_DATASET_PATH.exists():
        return pd.read_csv(LOCAL_DATASET_PATH)

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

    return df_encoded, encoders


def prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df_prepared = df.drop(columns=["id"]).copy()
    df_prepared["bmi"] = df_prepared["bmi"].fillna(df_prepared["bmi"].median())
    return _encode_categorical(df_prepared)


def load_dataset(
    test_size: float = 0.3,
    random_state: int = 42
):
    df = load_raw_dataframe()
    df_encoded, encoders = prepare_dataframe(df)

    X = df_encoded.drop(columns=["stroke"])
    y = df_encoded["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        "X_train": X_train.to_numpy(),
        "X_test": X_test.to_numpy(),
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "feature_names": X.columns.to_list(),
        "target_name": "stroke",
        "dataframe": df_encoded,
        "encoders": encoders,
    }
