from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import load_digits_dataset
from forest import OOBRandomForestClassifier, oob_accuracy_scorer
from metrics import (
    evaluate_model,
    plot_confusion_matrices,
    plot_grid_search,
    plot_importance_heatmaps,
    plot_metrics,
    plot_top_importances,
    plot_training_time,
)


LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
    cv_folds: int,
) -> tuple[GridSearchCV, float]:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    param_grid = {
        "n_estimators": [40, 80],
        "max_features": ["sqrt", 0.5],
        "max_depth": [None, 16],
        "min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(
        estimator=OOBRandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        # The scorer intentionally ignores the validation fold and uses estimator.oob_score_.
        scoring=oob_accuracy_scorer,
        cv=splitter,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )

    started = time.perf_counter()
    grid.fit(X_train, y_train)
    return grid, time.perf_counter() - started


def fit_with_timing(model, X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[object, float]:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - started


def make_sklearn_forest(best_params: dict, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        **best_params,
        criterion="gini",
        bootstrap=True,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1,
    )


def markdown_table(metrics: pd.DataFrame) -> list[str]:
    rows = [
        "| Model | Accuracy | Precision macro | Recall macro | F1 macro | OOB score | Train time, s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        rows.append(
            "| {model} | {accuracy:.4f} | {precision_macro:.4f} | {recall_macro:.4f} | "
            "{f1_macro:.4f} | {oob_score:.4f} | {train_time_sec:.4f} |".format(**row)
        )
    return rows


def make_report(
    metrics: pd.DataFrame,
    dataset_description: dict,
    best_params: dict,
    best_oob: float,
    grid_time: float,
    top_importance: pd.DataFrame,
) -> str:
    params = ", ".join(f"`{key}={value}`" for key, value in best_params.items())
    importance_lines = [
        "| Feature | OOB accuracy decrease | Std |",
        "|---|---:|---:|",
    ]
    for row in top_importance.head(8).to_dict("records"):
        importance_lines.append(
            "| {feature} | {importance_mean:.5f} | {importance_std:.5f} |".format(**row)
        )

    lines = [
        "# Experiment summary",
        "",
        f"Dataset: {dataset_description['name']}",
        f"Samples: {dataset_description['samples']}",
        f"Features: {dataset_description['features']} ({dataset_description['image_shape']} pixels)",
        f"Classes: {dataset_description['classes']}",
        "",
        "## Best custom RF parameters",
        "",
        params,
        f"Best OOB accuracy during grid search: `{best_oob:.4f}`",
        f"Grid search time: `{grid_time:.4f}` seconds",
        "",
        "## Metrics",
        "",
        *markdown_table(metrics),
        "",
        "## Top OOB^j importances",
        "",
        *importance_lines,
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 2: Random Forest with OOB model selection")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--grid-cv-folds", type=int, default=3)
    parser.add_argument("--importance-repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_digits_dataset(test_size=args.test_size, random_state=args.random_state)

    grid, grid_time = run_grid_search(
        bundle.X_train,
        bundle.y_train,
        args.random_state,
        args.grid_cv_folds,
    )
    best_params = dict(grid.best_params_)

    custom_model, custom_time = fit_with_timing(
        OOBRandomForestClassifier(**best_params, random_state=args.random_state),
        bundle.X_train,
        bundle.y_train,
    )

    importance = custom_model.compute_oob_permutation_importance(
        bundle.X_train,
        bundle.y_train,
        n_repeats=args.importance_repeats,
        random_state=args.random_state,
    )

    sklearn_model, sklearn_time = fit_with_timing(
        make_sklearn_forest(best_params, args.random_state),
        bundle.X_train,
        bundle.y_train,
    )

    metrics = pd.DataFrame(
        [
            evaluate_model("Custom Random Forest", custom_model, bundle.X_test, bundle.y_test, custom_time),
            evaluate_model("Sklearn RandomForest", sklearn_model, bundle.X_test, bundle.y_test, sklearn_time),
        ]
    )
    grid_results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")

    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)
    grid_results.to_csv(ARTIFACTS_DIR / "grid_search_results.csv", index=False)
    importance.to_csv(ARTIFACTS_DIR / "oob_feature_importance.csv", index=False)

    summary = {
        "dataset": bundle.description,
        "best_params": best_params,
        "best_oob_score": float(grid.best_score_),
        "grid_cv_folds": args.grid_cv_folds,
        "grid_time_sec": grid_time,
        "custom_train_time_sec": custom_time,
        "sklearn_train_time_sec": sklearn_time,
        "custom_oob_coverage": float(np.mean(custom_model.oob_counts_ > 0)),
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / "experiment_summary.md").write_text(
        make_report(metrics, bundle.description, best_params, grid.best_score_, grid_time, importance),
        encoding="utf-8",
    )

    plot_metrics(metrics, ARTIFACTS_DIR / "metrics.png")
    plot_training_time(metrics, grid_time, ARTIFACTS_DIR / "training_time.png")
    plot_confusion_matrices(
        {"Custom RF": custom_model, "Sklearn RF": sklearn_model},
        bundle.X_test,
        bundle.y_test,
        ARTIFACTS_DIR / "confusion_matrices.png",
    )
    plot_grid_search(grid_results, ARTIFACTS_DIR / "grid_search_oob.png")
    plot_importance_heatmaps(
        importance,
        sklearn_model.feature_importances_,
        ARTIFACTS_DIR / "feature_importance_heatmaps.png",
    )
    plot_top_importances(importance, ARTIFACTS_DIR / "top_oob_importances.png")

    report = make_report(metrics, bundle.description, best_params, grid.best_score_, grid_time, importance)
    print(report)
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
