import logging

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], list[bool]]:
    data = fetch_openml(data_id=25, as_frame=True, parser='auto')
    X_df: pd.DataFrame = data.data.copy()
    y_series = data.target

    # Бинаризуем: выжил → 0, погиб/усыплён → 1
    y = (y_series.astype(float) != 1).astype(int).values

    # Удаляем hospital_number — это идентификатор, а не признак
    X_df = X_df.drop(columns=['hospital_number'], errors='ignore')

    feature_names = list(X_df.columns)

    # Категориальные признаки определяем по dtype (fetch_openml помечает их как category)
    is_categorical = [X_df[col].dtype.name == 'category' for col in feature_names]

    # Кодируем категориальные признаки целыми числами, NaN оставляем как NaN
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
) -> float:

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    report = f"{title}\nAccuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    logging.info("\n" + report)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")

    return acc
