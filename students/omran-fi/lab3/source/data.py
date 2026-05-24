from __future__ import annotations

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "Survived"
DATASET = "yasserh/titanic-dataset"
DATA_FILE = "Titanic-Dataset.csv"

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "FarePerPerson",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "IsAlone",
]


def load_raw_titanic(project_root: Path) -> pd.DataFrame:
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    local_path = project_root / "data" / "titanic.csv"
    if local_path.exists():
        return pd.read_csv(local_path)

    downloaded_path = Path(kagglehub.dataset_download(DATASET)) / DATA_FILE
    df = pd.read_csv(downloaded_path)
    df.to_csv(local_path, index=False)
    return pd.read_csv(local_path)


def add_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    df["Title"] = df["Title"].replace(list(rare_titles), "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 1, "Yes", "No")

    df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)
    return df


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def load_features(project_root: Path):
    raw = add_titanic_features(load_raw_titanic(project_root))
    X = raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = raw[TARGET].astype(int).copy()
    dataset_info = {
        "n_samples": int(len(raw)),
        "positive_rate": float(y.mean()),
        "raw_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "missing_values": {
            k: int(v) for k, v in raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum().items()
        },
    }
    return X, y, dataset_info
