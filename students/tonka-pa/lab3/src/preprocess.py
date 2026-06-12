from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data") / "raw"

# Ordinal mappings for Diamonds categorical features
_CUT_ORDER = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
_COLOR_ORDER = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
_CLARITY_ORDER = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}


@dataclass
class DatasetBundle:

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    task: str
    description: str
    raw_shape: tuple[int, int]


def load_titanic() -> DatasetBundle:
    """
    Load Titanic (Kaggle competition) dataset from data/raw/titanic.csv.

    Classification target: survived (0/1).
    Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
    Missing values: Age (filled with median), Embarked (filled with mode).
    Categorical encoding: Sex (binary), Embarked (ordinal S/C/Q → 0/1/2).
    """
    path = RAW_DIR / "titanic.csv"
    df = pd.read_csv(path)
    raw_shape = df.shape

    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    target_col = "survived"

    df = df[feature_cols + [target_col]].copy()

    # impute
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # encode categoricals
    df["sex"] = (df["sex"] == "female").astype(int)
    df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(int)

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_cols,
        target_name=target_col,
        task="classification",
        description=(
            "Titanic (Kaggle): 891 passengers, 7 features, "
            "binary target — 0 not survived / 1 survived. "
            "Missing values in age and embarked handled by median/mode imputation."
        ),
        raw_shape=raw_shape,
    )


def load_diamonds() -> DatasetBundle:
    """
    Load Diamonds (Kaggle) dataset from data/raw/diamonds.csv.

    Regression target: price (USD).
    Features: carat, cut, color, clarity, depth, table, x, y, z.
    Categorical features encoded ordinally (natural quality ordering preserved).
    No missing values.
    """
    path = RAW_DIR / "diamonds.csv"
    df = pd.read_csv(path)
    raw_shape = df.shape

    feature_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    target_col = "price"

    df = df[feature_cols + [target_col]].copy()

    # ordinal encoding (natural quality order)
    df["cut"] = df["cut"].map(_CUT_ORDER)
    df["color"] = df["color"].map(_COLOR_ORDER)
    df["clarity"] = df["clarity"].map(_CLARITY_ORDER)

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_cols,
        target_name=target_col,
        task="regression",
        description=(
            "Diamonds (Kaggle / ggplot2): 53 940 diamonds, 9 features, "
            "continuous target — price in USD. Categorical features "
            "(cut, color, clarity) encoded ordinally by natural quality order."
        ),
        raw_shape=raw_shape,
    )


def load_data(task: str) -> DatasetBundle:
    """Return the dataset bundle for the requested task type."""
    if task == "classification":
        return load_titanic()
    if task == "regression":
        return load_diamonds()
    raise ValueError(f"Unsupported task: {task!r}")
