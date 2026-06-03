from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


CKD_UCI_ID = 336

NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
)

BINARY_FEATURES: dict[str, dict[str, int]] = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
}

def lab2_root() -> Path:
    return Path(__file__).resolve().parent.parent


def artifacts_dir() -> Path:
    d = lab2_root() / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean_str(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _encode_binary(values: pd.Series, mapping: dict[str, int]) -> pd.Series:
    cleaned = _clean_str(values)
    unknown = cleaned.dropna().loc[~cleaned.dropna().isin(mapping.keys())]
    assert unknown.empty, f"Неизвестные значения для {values.name}: {set(unknown)}"
    return cleaned.map(mapping).astype("float64")


def load_chronic_kidney_disease_uci(
    random_state: int = 42,
    test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ds = fetch_ucirepo(id=CKD_UCI_ID)
    X_frame = ds.data.features.copy()
    y_series = ds.data.targets.iloc[:, 0]

    y_clean = _clean_str(y_series).map({"ckd": 1, "notckd": 0})
    assert not y_clean.isna().any(), (
        f"Неожиданные значения цели: {set(_clean_str(y_series).unique())}"
    )
    y = y_clean.astype(np.int64).to_numpy()

    feature_names = list(NUMERIC_FEATURES) + list(BINARY_FEATURES.keys())
    n = len(X_frame)
    X = np.full((n, len(feature_names)), np.nan, dtype=np.float64)

    for j, c in enumerate(NUMERIC_FEATURES):
        col = pd.to_numeric(X_frame[c], errors="coerce")
        X[:, j] = col.to_numpy(dtype=np.float64)

    offset = len(NUMERIC_FEATURES)
    for k, (c, mapping) in enumerate(BINARY_FEATURES.items()):
        X[:, offset + k] = _encode_binary(X_frame[c], mapping).to_numpy(
            dtype=np.float64
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names
