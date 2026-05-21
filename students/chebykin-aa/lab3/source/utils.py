import logging

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from model import GradientBoostingClassifier as CustomGB


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

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    # Стратифицированная кросс-валидация, возвращает accuracy/precision/recall/f1 по фолдам
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for train_idx, val_idx in skf.split(X, y):
        gb = CustomGB(**params, random_state=random_state)
        gb.fit(X[train_idx], y[train_idx])
        y_pred = gb.predict(X[val_idx])

        acc = accuracy_score(y[val_idx], y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[val_idx], y_pred, average='weighted', zero_division=0
        )
        scores["accuracy"].append(acc)
        scores["precision"].append(precision)
        scores["recall"].append(recall)
        scores["f1"].append(f1)

    return {k: np.array(v) for k, v in scores.items()}

def save_cv_results(scores: dict[str, np.ndarray], title: str, save_path: str):
    lines = [f"{title}\n"]
    for metric, values in scores.items():
        lines.append(f"{metric}: {values.round(4)}  mean={values.mean():.4f} (+/- {values.std():.4f})\n")
    report = "".join(lines)
    logging.info("\n" + report)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
