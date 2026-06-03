import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.pipeline import Pipeline

from src.metrics import write_markdown_table
from src.visualization import plot_pruning_structure, plot_pruning_table


def select_best_ccp_alpha(
    model_factory: Callable[[float], Any],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str,
    output_dir: Path,
    prefix: str,
    max_candidates: int = 15,
) -> tuple[float, pd.DataFrame]:
    """Взял из примера sklearn."""
    table_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = _cost_complexity_path(model_factory(0.0), X_fit, y_fit)
    alphas = np.unique(np.asarray(path.ccp_alphas, dtype=float))
    if len(alphas) > 1:
        alphas = alphas[:-1]
    if len(alphas) > max_candidates:
        indexes = np.linspace(0, len(alphas) - 1, max_candidates, dtype=int)
        alphas = np.unique(alphas[indexes])
    if len(alphas) == 0:
        alphas = np.asarray([0.0])

    labels = np.unique(y_fit) if task == "classification" else None
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        started = time.perf_counter()
        model = model_factory(float(alpha))
        model.fit(X_fit, y_fit)
        fit_time = time.perf_counter() - started

        train_pred = model.predict(X_fit)
        val_pred = model.predict(X_val)
        train_score = _score(task, y_fit, train_pred, labels)
        val_score = _score(task, y_val, val_pred, labels)
        row = {
            "ccp_alpha": float(alpha),
            "train_score": float(train_score),
            "validation_score": float(val_score),
            "fit_time": float(fit_time),
            **_model_structure(model),
        }
        if task == "classification":
            row["validation_accuracy"] = float(accuracy_score(y_val, val_pred))
        rows.append(row)

    ranked = pd.DataFrame(rows).sort_values(
        ["validation_score", "ccp_alpha"],
        ascending=[False, True],
    )
    best_row = ranked.iloc[0]
    best_alpha = float(best_row["ccp_alpha"])
    table = pd.DataFrame(rows)
    table.to_csv(table_dir / f"{prefix}_ccp_alpha_search.csv", index=False)
    write_markdown_table(table, table_dir / f"{prefix}_ccp_alpha_search.md")
    plot_pruning_table(
        table,
        "validation_score",
        f"{prefix} pruning score",
        figures_dir / f"{prefix}_pruning_scores.png",
    )
    plot_pruning_structure(
        table,
        f"{prefix} pruning structure",
        figures_dir / f"{prefix}_pruning_structure.png",
    )
    return best_alpha, table


def _cost_complexity_path(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    if hasattr(model, "cost_complexity_pruning_path"):
        return model.cost_complexity_pruning_path(X, y)
    if isinstance(model, Pipeline):
        preprocessor = clone(model.named_steps["preprocessor"])
        tree = clone(model.named_steps["model"])
        transformed = preprocessor.fit_transform(X, y)
        return tree.cost_complexity_pruning_path(transformed, y)
    raise TypeError("Model does not provide a pruning path.")


def _score(
    task: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: np.ndarray | None,
) -> float:
    if task == "classification":
        if labels is not None and len(labels) == 2:
            return float(
                f1_score(y_true, y_pred, pos_label=labels[-1], zero_division=0)
            )
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return float(r2_score(y_true, y_pred))


def _model_structure(model: Any) -> dict[str, int]:
    if isinstance(model, Pipeline):
        tree = model.named_steps["model"]
        return {
            "depth": int(tree.get_depth()),
            "leaves": int(tree.get_n_leaves()),
            "node_count": int(tree.tree_.node_count),
        }
    return {
        "depth": int(model.get_depth()),
        "leaves": int(model.get_n_leaves()),
        "node_count": int(model.get_node_count()),
    }
