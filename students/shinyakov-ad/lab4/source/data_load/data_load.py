from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_PATH_STRING = "dongeorge/seed-from-uci"
DATASET_FILENAME = "Seed_Data.csv"
LOCAL_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / DATASET_FILENAME


def load_raw_dataframe() -> pd.DataFrame:
    if LOCAL_DATASET_PATH.exists():
        return pd.read_csv(LOCAL_DATASET_PATH)

    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(DATASET_PATH_STRING))
    return pd.read_csv(dataset_dir / DATASET_FILENAME)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_prepared = df.copy()
    if "Id" in df_prepared.columns:
        df_prepared = df_prepared.drop(columns=["Id"])
    return df_prepared.dropna().reset_index(drop=True)


def load_dataset(
    test_size: float = 0.3,
    random_state: int = 42,
    target_column: Optional[str] = "target",
):
    df = prepare_dataframe(load_raw_dataframe())

    if target_column is None:
        X = df
        X_train, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "feature_names": X.columns.to_list(),
            "target_name": target_column,
            "dataframe": df,
            "scaler": scaler,
        }
    else:
        X = df.drop(columns=[target_column])
        y = df[target_column]

    split = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_test, y_train, y_test = split

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "feature_names": X.columns.to_list(),
        "target_name": target_column,
        "dataframe": df,
        "scaler": scaler,
    }
