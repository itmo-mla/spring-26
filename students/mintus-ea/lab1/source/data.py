from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ADULT_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_FEATURES = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DatasetBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_types: dict[str, str]
    missing_summary: pd.Series
    class_balance: pd.Series
    source_name: str


def load_adult_income() -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series]:
    """Load Adult Income from UCI without storing the dataset in the repository."""
    with urlopen(ADULT_URL, timeout=30) as response:
        csv_text = response.read().decode("utf-8")

    frame = pd.read_csv(
        StringIO(csv_text),
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values=["?"],
    )
    frame = frame.dropna(subset=["income"]).reset_index(drop=True)

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("object")

    X = frame[FEATURES].copy()
    y = (frame["income"].str.strip() == ">50K").astype(int).to_numpy()

    missing_summary = X.isna().sum().sort_values(ascending=False)
    class_balance = pd.Series(y, name="income_gt_50k").value_counts(normalize=True).sort_index()
    return X, y, missing_summary, class_balance


def make_splits(
    sample_size: int = 6000,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    X, y, missing_summary, class_balance = load_adult_income()

    if sample_size and sample_size < len(X):
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            random_state=random_state,
            stratify=y,
        )
        X = X.reset_index(drop=True)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    feature_types = {
        **{name: "numeric" for name in NUMERIC_FEATURES},
        **{name: "categorical" for name in CATEGORICAL_FEATURES},
    }

    return DatasetBundle(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=np.asarray(y_train, dtype=int),
        y_val=np.asarray(y_val, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        feature_types=feature_types,
        missing_summary=missing_summary,
        class_balance=class_balance,
        source_name="UCI Adult Income",
    )
