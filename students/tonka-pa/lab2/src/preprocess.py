from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
RAW_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"
ZIP_PATH = RAW_DIR / "student_dropout.zip"
EXTRACT_DIR = RAW_DIR / "student_dropout"

CATEGORICAL_FEATURES = [
    "marital_status",
    "application_mode",
    "application_order",
    "course",
    "daytime/evening_attendance",
    "previous_qualification",
    "nacionality",
    "mother's_qualification",
    "father's_qualification",
    "mother's_occupation",
    "father's_occupation",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
]

NUMERIC_FEATURES = [
    "previous_qualification_(grade)",
    "admission_grade",
    "age_at_enrollment",
    "curricular_units_1st_sem_(credited)",
    "curricular_units_1st_sem_(enrolled)",
    "curricular_units_1st_sem_(evaluations)",
    "curricular_units_1st_sem_(approved)",
    "curricular_units_1st_sem_(grade)",
    "curricular_units_1st_sem_(without_evaluations)",
    "curricular_units_2nd_sem_(credited)",
    "curricular_units_2nd_sem_(enrolled)",
    "curricular_units_2nd_sem_(evaluations)",
    "curricular_units_2nd_sem_(approved)",
    "curricular_units_2nd_sem_(grade)",
    "curricular_units_2nd_sem_(without_evaluations)",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
]


@dataclass
class PreparedData:
    """Train/test arrays and metadata shared by all experiments."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    target_name: str
    raw_shape: tuple[int, int]
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    y_raw: pd.Series


def download_dataset(url: str = DATASET_URL, zip_path: Path = ZIP_PATH) -> Path:
    """Download the UCI dataset zip if it is not already cached."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        return zip_path

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    zip_path.write_bytes(response.content)
    return zip_path


def extract_dataset(zip_path: Path = ZIP_PATH, extract_dir: Path = EXTRACT_DIR) -> Path:
    """Extract the dataset zip and return the extracted directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def load_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and normalize the UCI student dropout CSV."""
    zip_path = download_dataset(zip_path=raw_dir / ZIP_PATH.name)
    extract_dir = extract_dataset(zip_path, raw_dir / EXTRACT_DIR.name)
    csv_paths = sorted(extract_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {extract_dir}.")

    data = _read_csv_with_delimiter(csv_paths[0])
    data.columns = normalize_column_names(data.columns)
    return data


def normalize_column_names(columns: pd.Index | list[str]) -> list[str]:
    """Normalize raw UCI column names to stable lowercase names."""
    normalized = []
    for column in columns:
        value = str(column).strip().lower()
        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"_+", "_", value)
        normalized.append(value)
    return normalized


def split_features_target(
    data: pd.DataFrame,
    target_name: str = "target",
) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target from the normalized dataframe."""
    if target_name not in data.columns:
        raise ValueError(f"Target column {target_name!r} was not found.")
    return data.drop(columns=[target_name]), data[target_name]


def detect_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return categorical and numeric feature names for preprocessing."""
    categorical = [name for name in CATEGORICAL_FEATURES if name in X.columns]
    numeric = [name for name in NUMERIC_FEATURES if name in X.columns]
    assigned = set(categorical) | set(numeric)

    missing_expected = (
        set(CATEGORICAL_FEATURES).union(NUMERIC_FEATURES).difference(X.columns)
    )
    if missing_expected:
        fallback_categorical, fallback_numeric = _fallback_feature_groups(
            X.drop(columns=list(assigned), errors="ignore")
        )
        categorical.extend(fallback_categorical)
        numeric.extend(fallback_numeric)
    else:
        remaining = [column for column in X.columns if column not in assigned]
        numeric.extend(remaining)

    return categorical, numeric


def build_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
) -> ColumnTransformer:
    """Build one-hot categorical and numeric passthrough preprocessing."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def prepare_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    """Download, preprocess, and split the dataset."""
    data = load_dataset()
    X, y = split_features_target(data)
    categorical_features, numeric_features = detect_feature_groups(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    preprocessor = build_preprocessor(categorical_features, numeric_features)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = _clean_transformed_feature_names(
        preprocessor.get_feature_names_out()
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train, columns=feature_names).to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False,
    )
    pd.DataFrame(X_test, columns=feature_names).to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False,
    )
    pd.Series(y_train, name="target").to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    pd.Series(y_test, name="target").to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    return PreparedData(
        X_train=np.asarray(X_train, dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        feature_names=feature_names,
        class_names=list(label_encoder.classes_),
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        target_name="target",
        raw_shape=data.shape,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        y_raw=y,
    )


def _read_csv_with_delimiter(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=";")
    if data.shape[1] == 1:
        data = pd.read_csv(path)
    return data


def _fallback_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical: list[str] = []
    numeric: list[str] = []
    for column in X.columns:
        series = X[column]
        if (
            pd.api.types.is_integer_dtype(series)
            and series.nunique(dropna=True) <= 20
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
        ):
            categorical.append(column)
        else:
            numeric.append(column)
    return categorical, numeric


def _clean_transformed_feature_names(names: Any) -> list[str]:
    cleaned = []
    for name in names:
        value = str(name)
        value = value.removeprefix("categorical__").removeprefix("numeric__")
        cleaned.append(value)
    return cleaned
