from __future__ import annotations

import atexit
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from boosting import BinaryGradientBoostingClassifier
from data import make_splits


LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"
IMAGES_DIR = LAB_DIR / "images"
RANDOM_STATE = 42
MODEL_PARAMS = {
    "n_estimators": 120,
    "learning_rate": 0.08,
    "max_depth": 3,
    "min_samples_leaf": 3,
    "subsample": 0.9,
    "random_state": RANDOM_STATE,
}


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, train_time: float) -> dict[str, Any]:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "train_time_seconds": train_time,
    }


def run_cross_validation(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "neg_log_loss": "neg_log_loss",
    }
    models = {
        "Custom Gradient Boosting": BinaryGradientBoostingClassifier(**MODEL_PARAMS),
        "sklearn GradientBoostingClassifier": GradientBoostingClassifier(**MODEL_PARAMS),
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        for fold_index in range(cv.get_n_splits()):
            rows.append(
                {
                    "model": name,
                    "fold": fold_index + 1,
                    "accuracy": scores["test_accuracy"][fold_index],
                    "roc_auc": scores["test_roc_auc"][fold_index],
                    "log_loss": -scores["test_neg_log_loss"][fold_index],
                    "fit_time_seconds": scores["fit_time"][fold_index],
                    "score_time_seconds": scores["score_time"][fold_index],
                }
            )
    return pd.DataFrame(rows)


def make_loss_curve(
    custom: BinaryGradientBoostingClassifier,
    sklearn_model: GradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for iteration, probabilities in enumerate(custom.staged_predict_proba(X_test), start=1):
        rows.append(
            {
                "model": "Custom Gradient Boosting",
                "iteration": iteration,
                "test_log_loss": log_loss(y_test, probabilities[:, 1], labels=[0, 1]),
            }
        )
    for iteration, probabilities in enumerate(sklearn_model.staged_predict_proba(X_test), start=1):
        rows.append(
            {
                "model": "sklearn GradientBoostingClassifier",
                "iteration": iteration,
                "test_log_loss": log_loss(y_test, probabilities[:, 1], labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    metrics: pd.DataFrame,
    cv_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    loss_curve: pd.DataFrame,
) -> None:
    mpl_config_dir = ARTIFACTS_DIR / ".mplconfig"
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    atexit.register(shutil.rmtree, mpl_config_dir, ignore_errors=True)
    import matplotlib.pyplot as plt

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_plot = metrics.set_index("model")[metric_columns].T
    ax = metric_plot.plot(kind="bar", figsize=(10, 5), ylim=(0.9, 1.0), rot=0)
    ax.set_title("Сравнение качества на тестовой выборке")
    ax.set_ylabel("Значение метрики")
    ax.set_xlabel("Метрика")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "metrics_comparison.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, group in loss_curve.groupby("model"):
        ax.plot(group["iteration"], group["test_log_loss"], label=model_name)
    ax.set_title("Test log loss по итерациям бустинга")
    ax.set_xlabel("Номер дерева")
    ax.set_ylabel("Log loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "loss_curve.png", dpi=160)
    plt.close()

    top_features = feature_importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features["feature"], top_features["custom_importance"], label="Custom")
    ax.barh(top_features["feature"], top_features["sklearn_importance"], alpha=0.65, label="sklearn")
    ax.set_title("Топ-10 признаков по важности")
    ax.set_xlabel("Importance")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_importance.png", dpi=160)
    plt.close()

    cv_plot = cv_summary.set_index("model")[["accuracy_mean", "roc_auc_mean", "log_loss_mean"]].T
    ax = cv_plot.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_title("Средние метрики 5-fold кросс-валидации")
    ax.set_ylabel("Значение")
    ax.set_xlabel("Метрика")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "cv_summary.png", dpi=160)
    plt.close()
    shutil.rmtree(mpl_config_dir, ignore_errors=True)


def run_experiment() -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data = make_splits(random_state=RANDOM_STATE)

    cv_results = run_cross_validation(data.X_train, data.y_train)
    cv_results.to_csv(ARTIFACTS_DIR / "cv_results.csv", index=False)

    custom = BinaryGradientBoostingClassifier(**MODEL_PARAMS)
    start = time.perf_counter()
    custom.fit(data.X_train, data.y_train)
    custom_train_time = time.perf_counter() - start
    custom_pred = custom.predict(data.X_test)
    custom_proba = custom.predict_proba(data.X_test)[:, 1]

    sklearn_model = GradientBoostingClassifier(**MODEL_PARAMS)
    start = time.perf_counter()
    sklearn_model.fit(data.X_train, data.y_train)
    sklearn_train_time = time.perf_counter() - start
    sklearn_pred = sklearn_model.predict(data.X_test)
    sklearn_proba = sklearn_model.predict_proba(data.X_test)[:, 1]

    metrics = pd.DataFrame(
        [
            evaluate_model("Custom Gradient Boosting", data.y_test, custom_pred, custom_proba, custom_train_time),
            evaluate_model(
                "sklearn GradientBoostingClassifier",
                data.y_test,
                sklearn_pred,
                sklearn_proba,
                sklearn_train_time,
            ),
        ]
    )
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": data.feature_names,
            "custom_importance": custom.feature_importances_,
            "sklearn_importance": sklearn_model.feature_importances_,
        }
    ).sort_values("custom_importance", ascending=False)
    feature_importance.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)

    loss_curve = make_loss_curve(custom, sklearn_model, data.X_test, data.y_test)
    loss_curve.to_csv(ARTIFACTS_DIR / "loss_curve.csv", index=False)

    cv_summary = (
        cv_results.groupby("model", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            log_loss_mean=("log_loss", "mean"),
            fit_time_mean_seconds=("fit_time_seconds", "mean"),
        )
        .sort_values("model")
    )
    cv_summary.to_csv(ARTIFACTS_DIR / "cv_summary.csv", index=False)
    save_plots(metrics, cv_summary, feature_importance, loss_curve)

    summary = {
        "dataset": data.source_name,
        "target_names": data.target_names,
        "sizes": {
            "train": len(data.X_train),
            "test": len(data.X_test),
            "features": len(data.feature_names),
        },
        "model_params": MODEL_PARAMS,
        "metrics": metrics.to_dict(orient="records"),
        "cross_validation_summary": cv_summary.to_dict(orient="records"),
        "confusion_matrices": {
            "custom": confusion_matrix(data.y_test, custom_pred).tolist(),
            "sklearn": confusion_matrix(data.y_test, sklearn_pred).tolist(),
        },
        "top_features": feature_importance.head(10).to_dict(orient="records"),
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run_experiment()
    metrics = pd.DataFrame(summary["metrics"])
    cv_summary = pd.DataFrame(summary["cross_validation_summary"])

    print(f"Dataset: {summary['dataset']}")
    print(f"Train/test sizes: {summary['sizes']['train']}/{summary['sizes']['test']}")
    print(f"Model params: {summary['model_params']}")
    print("\nTest metrics:")
    print(metrics.round(4).to_string(index=False))
    print("\nCross-validation summary:")
    print(cv_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
