import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.experiments.config import load_config
from src.experiments.pruning import select_best_ccp_alpha
from src.experiments.sklearn_baselines import (
    build_sklearn_pipeline,
    get_transformed_feature_names,
)
from src.metrics import (
    classification_metrics,
    regression_metrics,
    write_result_tables,
)
from src.preprocess import DatasetBundle, prepare_task_data
from src.tree import MyDecisionTreeClassifier, MyDecisionTreeRegressor
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_metric_comparison,
    plot_predicted_vs_true,
    plot_residuals,
    plot_roc_curves,
    plot_sklearn_tree_preview,
    plot_stats_bar,
)


RESULTS_DIR = Path("results") / "experiments"


def run_experiment(config_path: Path | str) -> pd.DataFrame:
    config = load_config(config_path)
    experiment_name = str(config["name"])
    output_dir = RESULTS_DIR / experiment_name
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    task = str(config["task"])
    bundle = prepare_task_data(
        task,
        drop_measured=bool(config["dataset"].get("drop_measured", True)),
    )
    split = config["split"]
    random_state = int(split.get("random_state", 42))
    model_params = _normalized_model_params(task, config["model"], random_state)

    stratify = bundle.y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=float(split.get("test_size", 0.25)),
        random_state=random_state,
        stratify=stratify,
    )

    selected_alphas = {"custom": model_params.get("ccp_alpha", 0.0), "sklearn": 0.0}
    if config["pruning"].get("enabled", False):
        selected_alphas = _select_pruning_alphas(
            config,
            bundle,
            X_train,
            y_train,
            model_params,
            output_dir,
        )

    custom_params = {**model_params, "ccp_alpha": selected_alphas["custom"]}
    sklearn_params = {**model_params, "ccp_alpha": selected_alphas["sklearn"]}
    custom_model = _build_custom_model(task, custom_params)
    sklearn_model = build_sklearn_pipeline(
        task,
        bundle.numeric_features,
        bundle.categorical_features,
        sklearn_params,
    )

    rows, predictions = _fit_and_evaluate_models(
        task,
        bundle,
        X_train,
        X_test,
        y_train,
        y_test,
        custom_model,
        sklearn_model,
        custom_params,
        sklearn_params,
    )

    metrics_frame = write_result_tables(rows, output_dir)
    _save_params(config, selected_alphas, model_params, output_dir)
    _save_plots(
        experiment_name,
        task,
        bundle,
        metrics_frame,
        predictions,
        custom_model,
        sklearn_model,
        figures_dir,
    )
    (tables_dir / f"{experiment_name}_custom_tree.txt").write_text(
        custom_model.export_text(max_depth=4),
        encoding="utf-8",
    )
    return metrics_frame


def _select_pruning_alphas(
    config: dict[str, Any],
    bundle: DatasetBundle,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict[str, Any],
    output_dir: Path,
) -> dict[str, float]:
    split = config["split"]
    task = str(config["task"])
    stratify = y_train if task == "classification" else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=float(split.get("validation_size", 0.25)),
        random_state=int(split.get("random_state", 42)),
        stratify=stratify,
    )
    base_params = {**model_params, "ccp_alpha": 0.0}
    max_candidates = int(config["pruning"].get("max_candidates", 15))

    custom_alpha, _ = select_best_ccp_alpha(
        lambda alpha: _build_custom_model(task, {**base_params, "ccp_alpha": alpha}),
        X_fit,
        y_fit,
        X_val,
        y_val,
        task,
        output_dir,
        "custom",
        max_candidates=max_candidates,
    )
    sklearn_alpha, _ = select_best_ccp_alpha(
        lambda alpha: build_sklearn_pipeline(
            task,
            bundle.numeric_features,
            bundle.categorical_features,
            {**base_params, "ccp_alpha": alpha},
        ),
        X_fit,
        y_fit,
        X_val,
        y_val,
        task,
        output_dir,
        "sklearn",
        max_candidates=max_candidates,
    )
    return {"custom": custom_alpha, "sklearn": sklearn_alpha}


def _fit_and_evaluate_models(
    task: str,
    bundle: DatasetBundle,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    custom_model: MyDecisionTreeClassifier | MyDecisionTreeRegressor,
    sklearn_model: Pipeline,
    custom_params: dict[str, Any],
    sklearn_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    for name, model, params in [
        ("custom", custom_model, custom_params),
        ("sklearn", sklearn_model, sklearn_params),
    ]:
        fit_started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - predict_started
        y_proba = model.predict_proba(X_test) if task == "classification" else None
        classes = _model_classes(model) if task == "classification" else None
        scores = (
            classification_metrics(y_test, y_pred, y_proba, classes)
            if task == "classification"
            else regression_metrics(y_test, y_pred)
        )
        row = {
            "model": name,
            "task": task,
            "parameters": json.dumps(params, ensure_ascii=False, default=str),
            **scores,
            "fit_time": fit_time,
            "predict_time": predict_time,
            **_model_structure(model),
        }
        rows.append(row)
        predictions[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "classes": classes,
        }
    predictions["y_test"] = y_test
    return rows, predictions


def _save_plots(
    experiment_name: str,
    task: str,
    bundle: DatasetBundle,
    metrics_frame: pd.DataFrame,
    predictions: dict[str, Any],
    custom_model: MyDecisionTreeClassifier | MyDecisionTreeRegressor,
    sklearn_model: Pipeline,
    figures_dir: Path,
) -> None:
    plot_metric_comparison(
        metrics_frame,
        task,
        figures_dir / f"{experiment_name}_metrics_comparison.png",
    )
    for column in ["fit_time", "predict_time", "depth", "leaves"]:
        plot_stats_bar(
            metrics_frame,
            column,
            figures_dir / f"{experiment_name}_{column}.png",
        )

    y_test = predictions["y_test"]
    model_predictions = {
        name: payload
        for name, payload in predictions.items()
        if isinstance(payload, dict) and "y_pred" in payload
    }

    if task == "classification":
        for name, payload in model_predictions.items():
            plot_confusion_matrix(
                y_test,
                payload["y_pred"],
                payload["classes"],
                figures_dir / f"{experiment_name}_confusion_matrix_{name}.png",
            )
        curves = {}
        for name, payload in model_predictions.items():
            model_labels = payload["classes"]
            if model_labels is not None and len(model_labels) == 2:
                positive_index = list(model_labels).index(model_labels[-1])
                curves[name] = (payload["y_proba"][:, positive_index], model_labels)
        plot_roc_curves(
            y_test, curves, figures_dir / f"{experiment_name}_roc_curve.png"
        )
        plot_feature_importances(
            list(bundle.X.columns),
            custom_model.feature_importances_,
            "Custom feature importances",
            figures_dir / f"{experiment_name}_feature_importance_custom.png",
        )
        _plot_sklearn_importances(
            sklearn_model,
            figures_dir / f"{experiment_name}_feature_importance_sklearn.png",
        )
    else:
        for name, payload in model_predictions.items():
            plot_predicted_vs_true(
                y_test,
                payload["y_pred"],
                f"{name} predicted vs true",
                figures_dir / f"{experiment_name}_predicted_vs_true_{name}.png",
            )
            plot_residuals(
                y_test,
                payload["y_pred"],
                f"{name} residuals",
                figures_dir / f"{experiment_name}_residuals_{name}.png",
            )
        plot_feature_importances(
            list(bundle.X.columns),
            custom_model.feature_importances_,
            "Custom feature importances",
            figures_dir / f"{experiment_name}_feature_importance_custom.png",
        )
        _plot_sklearn_importances(
            sklearn_model,
            figures_dir / f"{experiment_name}_feature_importance_sklearn.png",
        )

    try:
        sklearn_tree = sklearn_model.named_steps["model"]
        feature_names = get_transformed_feature_names(sklearn_model)
        plot_sklearn_tree_preview(
            sklearn_tree,
            feature_names,
            figures_dir / f"{experiment_name}_sklearn_tree_preview.png",
        )
    except Exception:
        pass


def _plot_sklearn_importances(sklearn_model: Pipeline, path: Path) -> None:
    tree = sklearn_model.named_steps["model"]
    names = get_transformed_feature_names(sklearn_model)
    plot_feature_importances(
        names, tree.feature_importances_, "Sklearn importances", path
    )


def _save_params(
    config: dict[str, Any],
    selected_alphas: dict[str, float],
    model_params: dict[str, Any],
    output_dir: Path,
) -> None:
    payload = {
        "config": config,
        "model_params": model_params,
        "selected_alphas": selected_alphas,
    }
    (output_dir / "params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _normalized_model_params(
    task: str,
    params: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    normalized = dict(params)
    normalized.setdefault("random_state", random_state)
    normalized.setdefault("ccp_alpha", 0.0)
    if task == "classification":
        normalized.setdefault("criterion", "gini")
    else:
        normalized.setdefault("criterion", "squared_error")
    return normalized


def _build_custom_model(
    task: str,
    params: dict[str, Any],
) -> MyDecisionTreeClassifier | MyDecisionTreeRegressor:
    if task == "classification":
        return MyDecisionTreeClassifier(**params)
    if task == "regression":
        return MyDecisionTreeRegressor(**params)
    raise ValueError(f"Unsupported task: {task}")


def _model_classes(model: Any) -> np.ndarray:
    if isinstance(model, Pipeline):
        return model.named_steps["model"].classes_
    return model.classes_


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
