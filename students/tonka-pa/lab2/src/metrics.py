from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fit_time: float,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray | None]:
    """Evaluate a fitted classifier and return metrics plus predictions."""
    started = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - started

    y_proba = _safe_predict_proba(model, X_test)
    metrics: dict[str, float | str] = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "roc_auc_ovr_weighted": _safe_roc_auc(y_test, y_proba),
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        "oob_score": _safe_float(getattr(model, "oob_score_", np.nan)),
    }
    metrics.update(model_tree_summary(model))
    return metrics, y_pred, y_proba


def model_tree_summary(model: Any) -> dict[str, float]:
    """Return mean tree depth and leaves for forests or one tree."""
    if hasattr(model, "estimators_"):
        trees = list(model.estimators_)
    elif hasattr(model, "tree_"):
        trees = [model]
    else:
        trees = []

    if not trees:
        return {"mean_tree_depth": np.nan, "mean_tree_leaves": np.nan}

    return {
        "mean_tree_depth": float(np.mean([tree.get_depth() for tree in trees])),
        "mean_tree_leaves": float(np.mean([tree.get_n_leaves() for tree in trees])),
    }


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    """Save a dataframe as CSV and simple Markdown."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a compact GitHub-style Markdown table without extra deps."""
    display = df if max_rows is None else df.head(max_rows)
    if display.empty:
        return "_No rows._\n"

    string_df = display.copy()
    for column in string_df.columns:
        string_df[column] = string_df[column].map(_format_value)

    headers = list(string_df.columns)
    rows = string_df.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _safe_predict_proba(model: Any, X_test: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X_test)
    except Exception:
        return None


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray | None) -> float:
    if y_proba is None:
        return np.nan
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        )
    except Exception:
        return np.nan


def _safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _format_value(value: Any) -> str:
    if isinstance(value, float | np.floating):
        if np.isnan(value):
            return "NaN"
        return f"{value:.4f}"
    return str(value)
