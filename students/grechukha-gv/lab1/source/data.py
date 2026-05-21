from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

NUM_FEATURES = [
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
]
CAT_FEATURES = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
FEATURE_NAMES = NUM_FEATURES + CAT_FEATURES


def lab1_root() -> Path:
    return Path(__file__).resolve().parent.parent


def artifacts_dir() -> Path:
    """Каталог артефактов запуска: графики и tree_rules.txt."""
    d = lab1_root() / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_chronic_kidney_disease(random_state=42):
    """
    Chronic Kidney Disease
    Источник: UCI ML Repository
    """
    ds = fetch_ucirepo(id=336)
    df = ds.data.features.copy()
    y_series = ds.data.targets["class"].astype(str).str.strip().map(
        {"ckd": 1, "notckd": 0}
    )
    valid = y_series.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y_series.loc[valid].astype(int).values

    for c in CAT_FEATURES:
        s = df[c].astype(str).str.strip()
        s = s.replace({"?": np.nan, "nan": np.nan, "NaN": np.nan})
        df[c] = s

    for c in NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df_features = df[FEATURE_NAMES]
    n = len(df_features)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=object)
    for j, c in enumerate(FEATURE_NAMES):
        X[:, j] = df_features[c].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, list(FEATURE_NAMES)
