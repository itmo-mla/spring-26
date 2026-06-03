"""Загрузка и подготовка данных."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

OPENML_CACHE_DIR = Path("data") / "raw" / "openml"


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    target_name: str
    task: str


def load_openml_1000() -> pd.DataFrame:
    OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(
        data_id=1000,
        as_frame=True,
        parser="auto",
        data_home=OPENML_CACHE_DIR,
    )
    if dataset.frame is not None:
        data = dataset.frame.copy()
    else:
        data = dataset.data.copy()
        data[dataset.target.name] = dataset.target
    return clean_missing_values(data)


def clean_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.replace("?", np.nan).copy()
    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            continue
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        non_missing = cleaned[column].notna()
        if non_missing.any() and converted[non_missing].notna().all():
            cleaned[column] = converted
    return cleaned


def prepare_task_data(
    task: str,
    drop_measured: bool = True,
) -> DatasetBundle:
    data = load_openml_1000()
    if task == "classification":
        target_name = "binaryClass"
        y = data[target_name]
        X = data.drop(columns=[target_name])
    elif task == "regression":
        target_name = "age"
        y = pd.to_numeric(data[target_name], errors="coerce")
        X = data.drop(columns=[target_name])
    else:
        raise ValueError(f"Unsupported task: {task}")

    if drop_measured:
        measured_columns = [
            column for column in X.columns if "measured" in str(column).lower()
        ]
        X = X.drop(columns=measured_columns)

    X = X.drop(columns=["TBG"])

    valid_target = y.notna()
    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    numeric_features, categorical_features = detect_feature_types(X)
    return DatasetBundle(
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_name=target_name,
        task=task,
    )


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for column in X.columns:
        if pd.api.types.is_bool_dtype(X[column]):
            categorical_features.append(column)
        elif pd.api.types.is_numeric_dtype(X[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)
    return numeric_features, categorical_features
