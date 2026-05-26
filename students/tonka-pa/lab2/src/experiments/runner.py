from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console

from src.experiments.config import load_config
from src.experiments.sklearn_baselines import (
    create_decision_tree,
    create_sklearn_random_forest,
)
from src.experiments.tuning import run_oob_grid_search
from src.metrics import evaluate_model, save_table
from src.preprocess import prepare_data
from src.random_forest import MyRandomForestClassifier
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_grid_search_summary,
    plot_metrics_comparison,
    plot_oob_comparison,
    plot_oob_curve,
    plot_oob_permutation_importance,
    plot_timing_comparison,
)

RESULTS_DIR = Path("results") / "experiments"
GRID_EXPERIMENT_DIR = RESULTS_DIR / "rf_oob_grid_search"
console = Console()


def run_experiment(config_path: str | Path) -> Path:
    """Run one experiment from a YAML config and return its output folder."""
    config = load_config(config_path)
    experiment_name = config["experiment_name"]
    experiment_dir = RESULTS_DIR / experiment_name
    figures_dir = experiment_dir / "figures"
    tables_dir = experiment_dir / "tables"
    custom_dir = experiment_dir / "custom"
    sklearn_dir = experiment_dir / "sklearn"
    for directory in [figures_dir, tables_dir, custom_dir, sklearn_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Running {experiment_name}[/bold]")
    data = prepare_data(
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
    )
    plot_class_distribution(data.y_raw, figures_dir / "class_distribution.png")

    custom_params = dict(config["model_params"])
    sklearn_params = dict(config["model_params"])
    if config["use_grid_search_best_params"]:
        custom_params = _load_best_params(
            GRID_EXPERIMENT_DIR / "best_params_custom.json",
            custom_params,
        )
        sklearn_params = _load_best_params(
            GRID_EXPERIMENT_DIR / "best_params_sklearn.json",
            sklearn_params,
        )

    grid_tables: dict[str, pd.DataFrame] = {}
    if config["run_grid_search"]:
        custom_params, sklearn_params, grid_tables = _run_grid_searches(
            config,
            data.X_train,
            data.y_train,
            experiment_dir,
            figures_dir,
        )

    metrics_rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}

    custom_model = MyRandomForestClassifier(
        **_with_random_state(custom_params, int(config["random_state"]))
    )
    custom_fit_time = _fit_with_timing(custom_model, data.X_train, data.y_train)
    row, y_pred, _ = evaluate_model(
        "custom_rf",
        custom_model,
        data.X_test,
        data.y_test,
        custom_fit_time,
    )
    metrics_rows.append(row)
    predictions["custom_rf"] = y_pred
    _save_feature_importances(
        custom_model.feature_importances_,
        data.feature_names,
        custom_dir / "impurity_feature_importance.csv",
        custom_dir / "impurity_feature_importance.md",
    )
    plot_feature_importance(
        custom_model.feature_importances_,
        data.feature_names,
        figures_dir / "custom_impurity_feature_importance.png",
        "Custom RF impurity feature importance",
    )
    custom_oob_curve = _compute_custom_oob_curve(
        custom_model,
        data.X_train,
        data.y_train,
    )
    if not custom_oob_curve.empty:
        save_table(
            custom_oob_curve,
            tables_dir / "custom_oob_score_vs_n_estimators.csv",
            tables_dir / "custom_oob_score_vs_n_estimators.md",
        )
        plot_oob_curve(
            custom_oob_curve,
            figures_dir / "custom_oob_score_vs_n_estimators.png",
            "Custom RF OOB score vs n_estimators",
        )

    if config["compute_oob_importance"]:
        oob_importance = custom_model.compute_oob_permutation_importance(
            data.X_train,
            data.y_train,
            feature_names=data.feature_names,
            random_state=int(config["random_state"]),
        )
        save_table(
            oob_importance,
            custom_dir / "oob_permutation_feature_importance.csv",
            custom_dir / "oob_permutation_feature_importance.md",
        )
        plot_oob_permutation_importance(
            oob_importance,
            figures_dir / "custom_oob_permutation_feature_importance.png",
        )

    sklearn_model = create_sklearn_random_forest(
        sklearn_params,
        random_state=int(config["random_state"]),
        n_jobs=sklearn_params.get("n_jobs"),
    )
    sklearn_fit_time = _fit_with_timing(sklearn_model, data.X_train, data.y_train)
    row, y_pred, _ = evaluate_model(
        "sklearn_rf",
        sklearn_model,
        data.X_test,
        data.y_test,
        sklearn_fit_time,
    )
    metrics_rows.append(row)
    predictions["sklearn_rf"] = y_pred
    _save_feature_importances(
        sklearn_model.feature_importances_,
        data.feature_names,
        sklearn_dir / "impurity_feature_importance.csv",
        sklearn_dir / "impurity_feature_importance.md",
    )
    plot_feature_importance(
        sklearn_model.feature_importances_,
        data.feature_names,
        figures_dir / "sklearn_impurity_feature_importance.png",
        "Sklearn RF impurity feature importance",
    )

    if config["include_decision_tree"]:
        tree_model = create_decision_tree(
            config["model_params"],
            random_state=int(config["random_state"]),
        )
        tree_fit_time = _fit_with_timing(tree_model, data.X_train, data.y_train)
        row, y_pred, _ = evaluate_model(
            "decision_tree",
            tree_model,
            data.X_test,
            data.y_test,
            tree_fit_time,
        )
        metrics_rows.append(row)
        predictions["decision_tree"] = y_pred

    metrics = pd.DataFrame(metrics_rows)
    save_table(metrics, experiment_dir / "metrics.csv", experiment_dir / "metrics.md")
    plot_confusion_matrix(
        data.y_test,
        predictions["custom_rf"],
        data.class_names,
        "Custom RF confusion matrix",
        figures_dir / "confusion_matrix_custom_rf.png",
    )
    plot_confusion_matrix(
        data.y_test,
        predictions["sklearn_rf"],
        data.class_names,
        "Sklearn RF confusion matrix",
        figures_dir / "confusion_matrix_sklearn_rf.png",
    )
    plot_metrics_comparison(metrics, figures_dir / "metrics_comparison.png")
    plot_timing_comparison(metrics, figures_dir / "timing_comparison.png")
    plot_oob_comparison(metrics, figures_dir / "oob_score_comparison.png")

    _save_params(
        experiment_dir / "params.json",
        config,
        custom_params,
        sklearn_params,
        data,
        grid_tables,
    )
    console.print(f"[green]Saved artifacts to {experiment_dir}[/green]")
    return experiment_dir


def _run_grid_searches(
    config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    experiment_dir: Path,
    figures_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, pd.DataFrame]]:
    random_state = int(config["random_state"])
    base_params = dict(config["model_params"])

    console.print("Running OOB grid search for custom RF")
    best_custom, custom_results = run_oob_grid_search(
        lambda params: MyRandomForestClassifier(
            **_with_random_state(params, random_state)
        ),
        X_train,
        y_train,
        config["param_grid"],
        base_params=base_params,
    )
    console.print("Running OOB grid search for sklearn RF")
    best_sklearn, sklearn_results = run_oob_grid_search(
        lambda params: create_sklearn_random_forest(
            params,
            random_state=random_state,
            n_jobs=params.get("n_jobs"),
        ),
        X_train,
        y_train,
        config["param_grid"],
        base_params=base_params,
    )

    save_table(
        custom_results,
        experiment_dir / "grid_search_results_custom.csv",
        experiment_dir / "grid_search_results_custom.md",
    )
    save_table(
        sklearn_results,
        experiment_dir / "grid_search_results_sklearn.csv",
        experiment_dir / "grid_search_results_sklearn.md",
    )
    _write_json(experiment_dir / "best_params_custom.json", best_custom)
    _write_json(experiment_dir / "best_params_sklearn.json", best_sklearn)

    plot_oob_curve(
        custom_results,
        figures_dir / "custom_grid_oob_score_vs_n_estimators.png",
        "Custom RF grid-search OOB score",
    )
    plot_oob_curve(
        sklearn_results,
        figures_dir / "sklearn_grid_oob_score_vs_n_estimators.png",
        "Sklearn RF grid-search OOB score",
    )
    plot_grid_search_summary(
        custom_results,
        figures_dir / "custom_grid_search_heatmap.png",
        "Custom RF grid-search summary",
    )
    plot_grid_search_summary(
        sklearn_results,
        figures_dir / "sklearn_grid_search_heatmap.png",
        "Sklearn RF grid-search summary",
    )
    return (
        best_custom,
        best_sklearn,
        {
            "custom_grid_search": custom_results,
            "sklearn_grid_search": sklearn_results,
        },
    )


def _fit_with_timing(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> float:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    return time.perf_counter() - started


def _save_feature_importances(
    importances: np.ndarray,
    feature_names: list[str],
    csv_path: Path,
    md_path: Path,
) -> None:
    data = pd.DataFrame({"feature": feature_names, "importance": importances})
    data = data.sort_values("importance", ascending=False).reset_index(drop=True)
    save_table(data, csv_path, md_path)


def _compute_custom_oob_curve(
    model: MyRandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    if not getattr(model, "bootstrap", False):
        return pd.DataFrame()

    n_estimators = len(model.estimators_)
    checkpoints = np.unique(
        np.linspace(1, n_estimators, num=min(10, n_estimators), dtype=int)
    )
    rows = []
    for checkpoint in checkpoints:
        proba_sum = np.zeros((X_train.shape[0], model.n_classes_), dtype=float)
        counts = np.zeros(X_train.shape[0], dtype=int)
        for tree, oob_indices in zip(
            model.estimators_[:checkpoint],
            model.estimators_oob_indices_[:checkpoint],
            strict=True,
        ):
            if len(oob_indices) == 0:
                continue
            proba_sum[oob_indices] += model._aligned_tree_proba(  # noqa: SLF001
                tree,
                X_train[oob_indices],
            )
            counts[oob_indices] += 1

        available = counts > 0
        if not np.any(available):
            score = np.nan
        else:
            predictions = model.classes_[
                np.argmax(proba_sum[available] / counts[available, None], axis=1)
            ]
            score = float(np.mean(predictions == y_train[available]))
        rows.append({"n_estimators": int(checkpoint), "oob_score": score})
    return pd.DataFrame(rows)


def _save_params(
    path: Path,
    config: dict[str, Any],
    custom_params: dict[str, Any],
    sklearn_params: dict[str, Any],
    data: Any,
    grid_tables: dict[str, pd.DataFrame],
) -> None:
    payload = {
        "config": config,
        "custom_model_params": custom_params,
        "sklearn_model_params": sklearn_params,
        "dataset": {
            "raw_shape": data.raw_shape,
            "target_name": data.target_name,
            "class_names": data.class_names,
            "n_transformed_features": len(data.feature_names),
            "categorical_features": data.categorical_features,
            "numeric_features": data.numeric_features,
        },
        "grid_search_rows": {name: len(table) for name, table in grid_tables.items()},
    }
    _write_json(path, payload)


def _load_best_params(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(fallback)
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _with_random_state(params: dict[str, Any], random_state: int) -> dict[str, Any]:
    merged = dict(params)
    merged["random_state"] = random_state
    return merged


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
