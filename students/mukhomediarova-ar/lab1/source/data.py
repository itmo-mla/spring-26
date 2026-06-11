from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "survived"


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
    source_name: str


def load_titanic() -> tuple[pd.DataFrame, np.ndarray, pd.Series, str]:
    """Load Titanic from OpenML; fall back to a deterministic local sample offline."""
    try:
        bunch = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
        frame = bunch.frame.copy()
        source_name = "OpenML Titanic"
        if TARGET not in frame.columns:
            frame[TARGET] = bunch.target
    except Exception:
        frame = _make_local_titanic_like_sample()
        source_name = "local Titanic-like fallback"

    frame = frame[FEATURES + [TARGET]].copy()
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("object")

    y = pd.to_numeric(frame[TARGET], errors="coerce").astype(int).to_numpy()
    X = frame[FEATURES]
    return X, y, X.isna().sum().sort_values(ascending=False), source_name


def make_splits(
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    X, y, missing_summary, source_name = load_titanic()

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
        **{feature: "numeric" for feature in NUMERIC_FEATURES},
        **{feature: "categorical" for feature in CATEGORICAL_FEATURES},
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
        source_name=source_name,
    )


def _make_local_titanic_like_sample(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    pclass = rng.choice([1, 2, 3], size=n_samples, p=[0.24, 0.22, 0.54])
    sex = rng.choice(["female", "male"], size=n_samples, p=[0.36, 0.64])
    embarked = rng.choice(["S", "C", "Q"], size=n_samples, p=[0.72, 0.18, 0.10]).astype(object)
    age = np.clip(rng.normal(30, 14, size=n_samples), 0.5, 80)
    sibsp = rng.poisson(0.5, size=n_samples)
    parch = rng.poisson(0.35, size=n_samples)
    fare = np.maximum(
        rng.gamma(shape=2.0, scale=12.0, size=n_samples) * (4 - pclass) + rng.normal(0, 4, n_samples),
        3,
    )

    logits = (
        -1.2
        + 1.9 * (sex == "female")
        + 0.65 * (pclass == 1)
        + 0.25 * (pclass == 2)
        - 0.018 * np.maximum(age - 12, 0)
        + 0.75 * (age < 12)
        + 0.22 * (embarked == "C")
        - 0.12 * sibsp
        - 0.08 * parch
    )
    probabilities = 1 / (1 + np.exp(-logits))
    survived = rng.binomial(1, probabilities)

    frame = pd.DataFrame(
        {
            "pclass": pclass,
            "age": age.round(1),
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare.round(2),
            "sex": sex,
            "embarked": embarked,
            "survived": survived,
        }
    )

    age_missing = rng.random(n_samples) < 0.18
    fare_missing = rng.random(n_samples) < 0.04
    embarked_missing = rng.random(n_samples) < 0.03
    frame.loc[age_missing, "age"] = np.nan
    frame.loc[fare_missing, "fare"] = np.nan
    frame.loc[embarked_missing, "embarked"] = np.nan
    return frame
