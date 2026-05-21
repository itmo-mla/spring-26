from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
import pandas as pd
from data_load import load_dataset
from model import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from visualization import plot_cv_metrics, plot_fit_time


RANDOM_STATE = 42
N_SPLITS = 5


def _predict_positive_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    return np.asarray(model.predict(X), dtype=float)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics["roc_auc"] = np.nan

    return metrics


def cross_validate_model(
    name: str,
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict]:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        model = model_factory()
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        started_at = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - started_at

        y_pred = np.asarray(model.predict(X_valid))
        y_proba = _predict_positive_proba(model, X_valid)
        metrics = evaluate_predictions(y_valid, y_pred, y_proba)
        rows.append({"model": name, "fold": fold_idx, "fit_time_sec": fit_time, **metrics})

    return rows


def summarize_cv(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["fit_time_sec", "accuracy", "precision", "recall", "f1", "roc_auc"]
    return (
        fold_results.groupby("model", as_index=False)[metric_cols]
        .mean()
        .sort_values("model")
        .reset_index(drop=True)
    )


def main() -> None:
    lab_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = lab_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(test_size=0.3, random_state=RANDOM_STATE)
    X = np.concatenate([dataset["X_train"], dataset["X_test"]])
    y = np.concatenate([dataset["y_train"], dataset["y_test"]])

    model_factories: list[tuple[str, Callable[[], object]]] = [
        (
            "custom_gradient_boosting",
            lambda: GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "sklearn_gradient_boosting",
            lambda: SklearnGradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    fold_rows = []
    for name, factory in model_factories:
        fold_rows.extend(cross_validate_model(name, factory, X, y))

    if not fold_rows:
        raise RuntimeError("No models were evaluated.")

    fold_results = pd.DataFrame(fold_rows)
    summary = summarize_cv(fold_results)

    plot_cv_metrics(summary, artifacts_dir / "cv_metrics.png")
    plot_fit_time(summary, artifacts_dir / "fit_time.png")

    print("\nCross-validation summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nSaved artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
