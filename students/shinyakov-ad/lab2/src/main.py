from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data_load import load_dataset
from model import RandomForestRegressorCustom
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def evaluate_regressor(model, X, y):
    prediction = np.asarray(model.predict(X), dtype=float)

    mse = mean_squared_error(y, prediction)
    metrics = {
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y, prediction),
    }
    return metrics, prediction


def print_metrics(name, metrics):
    print(f"\n{name}")
    for key in ["rmse", "r2"]:
        print(f"  {key}: {metrics[key]:.4f}")


def oob_r2_scorer(estimator, X, y):
    if not hasattr(estimator, "oob_score_"):
        return np.nan
    return estimator.oob_score_


def train_with_grid_search(X_train, y_train):
    param_grid = {
        "n_estimators": [20, 40],
        "max_depth": [8, 12],
        "max_features": ["sqrt", 0.7],
        "min_samples_leaf": [1, 3],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressorCustom(random_state=42),
        param_grid=param_grid,
        scoring=oob_r2_scorer,
        cv=[(np.arange(len(X_train)), np.arange(len(X_train)))],
        n_jobs=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def plot_metrics(results, output_path):
    labels = [row["model"] for row in results]
    x = np.arange(len(labels))
    width = 0.35

    rmse = np.array([row["rmse"] for row in results], dtype=float)
    r2 = np.array([row["r2"] for row in results], dtype=float)
    train_time = np.array([row["train_time"] for row in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_left, ax_right = axes

    ax_left.bar(x, rmse, width=0.5, label="RMSE")
    ax_left.set_title("RMSE (test)")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.grid(axis="y", alpha=0.3)
    ax_left.legend()

    ax_right.bar(x - width / 2, r2, width=width, label="R2")
    ax_right.bar(x + width / 2, train_time, width=width, label="train_time_sec")
    ax_right.set_title("Quality and Training Time")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(labels)
    ax_right.grid(axis="y", alpha=0.3)
    ax_right.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(feature_names, importances, output_path):
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[idx] for idx in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_names, sorted_importances)
    ax.invert_yaxis()
    ax.set_title("OOB Permutation Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, df = load_dataset()
    feature_names = [column for column in df.columns if column != "Median_House_Value"]

    grid_search = train_with_grid_search(X_train, y_train)
    custom_forest = grid_search.best_estimator_

    start = time.perf_counter()
    custom_forest.fit(X_train, y_train)
    custom_train_time = time.perf_counter() - start

    custom_metrics, _ = evaluate_regressor(custom_forest, X_test, y_test)
    custom_importance = custom_forest.compute_oob_feature_importance(X_train, y_train)

    start = time.perf_counter()
    sklearn_forest = RandomForestRegressor(**grid_search.best_params_, bootstrap=True, oob_score=True, random_state=42)
    sklearn_forest.fit(X_train, y_train)
    sklearn_train_time = time.perf_counter() - start

    sklearn_metrics, _ = evaluate_regressor(sklearn_forest, X_test, y_test)

    print("Best params from grid search:")
    print(grid_search.best_params_)
    print(f"Best OOB R2: {grid_search.best_score_:.4f}")
    print_metrics("Custom random forest", custom_metrics)
    print_metrics("Sklearn random forest", sklearn_metrics)

    results = [
        {"model": "custom_rf", "train_time": custom_train_time, **custom_metrics},
        {"model": "sklearn_rf", "train_time": sklearn_train_time, **sklearn_metrics},
    ]

    print("\nSummary (test):")
    print("model         rmse        r2         time_sec")
    for row in results:
        print(
            f"{row['model']:<13} "
            f"{row['rmse']:<11.4f} "
            f"{row['r2']:<10.4f} "
            f"{row['train_time']:.4f}"
        )

    print("\nTop-10 OOB feature importance:")
    top_indices = np.argsort(custom_importance)[::-1][:10]
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {custom_importance[idx]:.6f}")

    plot_metrics(results, artifacts_dir / "regression_metrics_test.png")
    plot_feature_importance(
        feature_names,
        custom_importance,
        artifacts_dir / "feature_importance_oob.png",
    )

    print(f"Saved plots: {artifacts_dir / 'regression_metrics_test.png'}")
    print(f"Saved plots: {artifacts_dir / 'feature_importance_oob.png'}")


if __name__ == "__main__":
    main()
