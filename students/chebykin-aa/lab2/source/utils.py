import logging

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid

from model import RandomForestClassifier as CustomRF


def load_data():
    data = fetch_openml(data_id=25, as_frame=True, parser='auto')
    X_df: pd.DataFrame = data.data.copy()
    y_series = data.target

    # Бинаризуем метки классов
    y = (y_series.astype(float) != 1).astype(int).values

    # Удаляем hospital_number, т.к. это не признак
    X_df = X_df.drop(columns=['hospital_number'], errors='ignore')

    feature_names = list(X_df.columns)

    # Категориальные признаки определяем по dtype
    is_categorical = [X_df[col].dtype.name == 'category' for col in feature_names]

    # Кодируем категориальные признаки целыми числами, NaN пропускаем
    X = np.full((len(X_df), len(feature_names)), np.nan, dtype=float)
    for i, col in enumerate(feature_names):
        if is_categorical[i]:
            codes = pd.Categorical(X_df[col]).codes.astype(float)
            codes[X_df[col].isna()] = np.nan
            X[:, i] = codes
        else:
            X[:, i] = pd.to_numeric(X_df[col], errors='coerce').values

    return X, y, feature_names, is_categorical


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
    save_path: str | None = None,
):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    report = (
        f"{title}\n"
        f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )
    logging.info("\n" + report)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")

    return acc


def oob_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    random_state: int = 42,
):
    best_params: dict = {}
    best_oob: float = -1.0

    grid = list(ParameterGrid(param_grid))
    logging.info(f"Grid search: {len(grid)} комбинаций")

    for params in grid:
        rf = CustomRF(**params, random_state=random_state)
        rf.fit(X_train, y_train)
        oob = rf.oob_score_
        logging.info(f"params={params}  OOB={oob:.4f}")
        if oob > best_oob:
            best_oob = oob
            best_params = params

    return best_params, best_oob
