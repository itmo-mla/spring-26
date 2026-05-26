import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split

from src.boosting import MyGradientBoostingClassifier, MyGradientBoostingRegressor
from src.experiments.config import load_config
from src.experiments.sklearn_baselines import build_sklearn_model
from src.metrics import classification_metrics, regression_metrics, write_result_tables
from src.preprocess import DatasetBundle, load_data
from src.visualization import (
    plot_confusion_matrix,
    plot_cv_comparison,
    plot_feature_importances,
    plot_learning_curve,
    plot_metric_comparison,
    plot_predicted_vs_true,
    plot_residuals,
    plot_roc_curves,
    plot_stats_bar,
)


RESULTS_DIR = Path("results") / "experiments"


def run_experiment(config_path: Path | str) -> pd.DataFrame:
    """Run one configured experiment and save all artifacts."""
    config = load_config(config_path)
    name = str(config["name"])
    task = str(config["task"])
    output_dir = RESULTS_DIR / name
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_data(task)
    split = config["split"]
    random_state = int(split.get("random_state", 42))
    cv_folds = int(split.get("cv_folds", 5))
    model_params = _normalize_params(config["model"], random_state)

    stratify = bundle.y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=float(split.get("test_size", 0.2)),
        random_state=random_state,
        stratify=stratify,
    )

    custom_model = _build_custom_model(task, model_params)
    sklearn_model = build_sklearn_model(task, model_params)

    rows, predictions = _fit_and_evaluate(
        task,
        X_train,
        X_test,
        y_train,
        y_test,
        custom_model,
        sklearn_model,
        model_params,
    )

    cv_rows = _cross_validate(task, bundle, model_params, cv_folds, random_state)

    metrics_frame = write_result_tables(rows, output_dir)
    _save_cv_table(cv_rows, output_dir)
    _save_params(config, model_params, output_dir)
    _save_plots(
        name,
        task,
        bundle,
        metrics_frame,
        cv_rows,
        predictions,
        custom_model,
        sklearn_model,
        figures_dir,
    )
    return metrics_frame


def _fit_and_evaluate(
    task: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    custom_model: Any,
    sklearn_model: Any,
    model_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}

    for name, model in [("custom", custom_model), ("sklearn", sklearn_model)]:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - t1

        y_proba = model.predict_proba(X_test) if task == "classification" else None

        if task == "classification":
            scores = classification_metrics(y_test, y_pred, y_proba)
        else:
            scores = regression_metrics(y_test, y_pred)

        rows.append(
            {
                "model": name,
                "task": task,
                "parameters": json.dumps(model_params, default=str),
                **scores,
                "fit_time": fit_time,
                "predict_time": predict_time,
            }
        )
        predictions[name] = {"y_pred": y_pred, "y_proba": y_proba, "model": model}

    predictions["y_test"] = y_test
    return rows, predictions


def _cross_validate(
    task: str,
    bundle: DatasetBundle,
    model_params: dict[str, Any],
    cv_folds: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if task == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = "accuracy"
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = "r2"

    cv_rows = []
    for label, model in [
        ("custom", _build_custom_model(task, model_params)),
        ("sklearn", build_sklearn_model(task, model_params)),
    ]:
        scores = cross_val_score(model, bundle.X, bundle.y, cv=cv, scoring=scoring)
        cv_rows.append(
            {
                "model": label,
                "cv_mean": float(np.mean(scores)),
                "cv_std": float(np.std(scores)),
                "cv_scores": scores.tolist(),
                "scoring": scoring,
            }
        )
    return cv_rows


def _save_plots(
    experiment_name: str,
    task: str,
    bundle: DatasetBundle,
    metrics_frame: pd.DataFrame,
    cv_rows: list[dict],
    predictions: dict[str, Any],
    custom_model: Any,
    sklearn_model: Any,
    figures_dir: Path,
) -> None:
    plot_metric_comparison(
        metrics_frame,
        task,
        figures_dir / f"{experiment_name}_metrics_comparison.png",
    )
    plot_cv_comparison(
        cv_rows,
        task,
        figures_dir / f"{experiment_name}_cv_comparison.png",
    )
    for col in ["fit_time", "predict_time"]:
        plot_stats_bar(metrics_frame, col, figures_dir / f"{experiment_name}_{col}.png")

    # learning curves from custom model
    if hasattr(custom_model, "train_loss_") and custom_model.train_loss_:
        plot_learning_curve(
            custom_model.train_loss_,
            "Custom model training loss",
            figures_dir / f"{experiment_name}_learning_curve_custom.png",
        )
    if hasattr(sklearn_model, "train_score_"):
        # sklearn GBM stores train_score_ (deviance/loss per iteration)
        plot_learning_curve(
            sklearn_model.train_score_.tolist(),
            "Sklearn model training loss",
            figures_dir / f"{experiment_name}_learning_curve_sklearn.png",
        )

    y_test = predictions["y_test"]
    model_preds = {k: v for k, v in predictions.items() if isinstance(v, dict)}

    if task == "classification":
        for name, payload in model_preds.items():
            plot_confusion_matrix(
                y_test,
                payload["y_pred"],
                np.array([0, 1]),
                figures_dir / f"{experiment_name}_confusion_matrix_{name}.png",
            )
        roc_curves = {}
        for name, payload in model_preds.items():
            if payload["y_proba"] is not None:
                roc_curves[name] = payload["y_proba"][:, 1]
        plot_roc_curves(y_test, roc_curves, figures_dir / f"{experiment_name}_roc_curve.png")
        plot_feature_importances(
            bundle.feature_names,
            custom_model.feature_importances_,
            "Custom feature importances",
            figures_dir / f"{experiment_name}_feature_importance_custom.png",
        )
        plot_feature_importances(
            bundle.feature_names,
            sklearn_model.feature_importances_,
            "Sklearn feature importances",
            figures_dir / f"{experiment_name}_feature_importance_sklearn.png",
        )
    else:
        for name, payload in model_preds.items():
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
            bundle.feature_names,
            custom_model.feature_importances_,
            "Custom feature importances",
            figures_dir / f"{experiment_name}_feature_importance_custom.png",
        )
        plot_feature_importances(
            bundle.feature_names,
            sklearn_model.feature_importances_,
            "Sklearn feature importances",
            figures_dir / f"{experiment_name}_feature_importance_sklearn.png",
        )


def _save_cv_table(cv_rows: list[dict], output_dir: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "model": r["model"],
                "scoring": r["scoring"],
                "cv_mean": r["cv_mean"],
                "cv_std": r["cv_std"],
            }
            for r in cv_rows
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "cv_scores.csv", index=False)
    lines = [
        "| model | scoring | cv_mean | cv_std |",
        "| --- | --- | --- | --- |",
    ]
    for r in cv_rows:
        lines.append(f"| {r['model']} | {r['scoring']} | {r['cv_mean']:.6g} | {r['cv_std']:.6g} |")
    (output_dir / "cv_scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_params(
    config: dict[str, Any],
    model_params: dict[str, Any],
    output_dir: Path,
) -> None:
    payload = {"config": config, "model_params": model_params}
    (output_dir / "params.json").write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _normalize_params(params: dict[str, Any], random_state: int) -> dict[str, Any]:
    out = dict(params)
    out.setdefault("n_estimators", 100)
    out.setdefault("learning_rate", 0.1)
    out.setdefault("max_depth", 3)
    out.setdefault("min_samples_split", 2)
    out.setdefault("min_samples_leaf", 1)
    out.setdefault("subsample", 1.0)
    out.setdefault("random_state", random_state)
    return out


def _build_custom_model(
    task: str,
    params: dict[str, Any],
) -> MyGradientBoostingClassifier | MyGradientBoostingRegressor:
    allowed = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "subsample",
        "random_state",
    }
    p = {k: v for k, v in params.items() if k in allowed}
    if task == "classification":
        return MyGradientBoostingClassifier(**p)
    if task == "regression":
        return MyGradientBoostingRegressor(**p)
    raise ValueError(f"Unsupported task: {task!r}")
