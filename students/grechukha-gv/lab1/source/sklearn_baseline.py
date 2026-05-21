"""Метрики классификации и эталон sklearn DecisionTreeClassifier"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from data import CAT_FEATURES, NUM_FEATURES


def metrics_report(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_metrics(name, m):
    print(
        f"{name:22} | Acc: {m['accuracy']:.4f} | P: {m['precision']:.4f} | "
        f"R: {m['recall']:.4f} | F1: {m['f1']:.4f}"
    )


def _arrays_to_df(X, feature_names):
    df = pd.DataFrame(np.asarray(X), columns=feature_names)
    for c in NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str).replace("nan", np.nan)
    return df


def run_sklearn_reference(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Эталон: импутация + One-Hot категорий + DecisionTreeClassifier(gini).
    Возвращает метрики на val/test и предсказания на val/test для матриц ошибок
    """
    df_train = _arrays_to_df(X_train, feature_names)
    df_val = _arrays_to_df(X_val, feature_names)
    df_test = _arrays_to_df(X_test, feature_names)

    try:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
            ("cat", ohe, CAT_FEATURES),
        ]
    )
    X_tr = preprocessor.fit_transform(df_train)
    X_va = preprocessor.transform(df_val)
    X_te = preprocessor.transform(df_test)

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        random_state=42,
    )
    clf.fit(X_tr, y_train)
    pred_val = clf.predict(X_va)
    pred_test = clf.predict(X_te)
    return (
        metrics_report(y_val, pred_val),
        metrics_report(y_test, pred_test),
        pred_val,
        pred_test,
    )


def confusion_breakdown(y_true, y_pred, positive_label=1):
    """TN, FP, FN, TP для бинарной задачи (положительный класс = positive_label)"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
