"""Metrics and result-table utilities for density and classification experiments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def density_metrics(model, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> dict:
    """Per-sample average log-likelihood + BIC/AIC on each split."""
    return {
        "log_lik_train": float(model.score(X_train)),
        "log_lik_val": float(model.score(X_val)),
        "log_lik_test": float(model.score(X_test)),
        "bic_train": float(model.bic(X_train)),
        "aic_train": float(model.aic(X_train)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Binary classification metrics. ``y_score`` is the positive-class probability."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[int] | None = None) -> pd.DataFrame:
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def rows_to_markdown(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False, floatfmt=".5f")


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
