from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from data import make_splits
from forest import OOBRandomForestClassifier, oob_accuracy_scorer


matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"
IMAGES_DIR = LAB_DIR / "images"
RANDOM_STATE = 42


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, train_time: float) -> dict[str, float | str]:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "train_time_seconds": train_time,
    }


def run_oob_grid_search(X_train: pd.DataFrame, y_train: np.ndarray) -> GridSearchCV:
    param_grid = {
        "n_estimators": [40, 80, 120],
        "max_features": ["sqrt", "log2", 0.5],
        "max_depth": [None, 5, 8],
        "min_samples_leaf": [1, 3],
    }
    train_indices = np.arange(len(y_train))
    model = OOBRandomForestClassifier(random_state=RANDOM_STATE, bootstrap=True)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=oob_accuracy_scorer,
        cv=[(train_indices, train_indices)],
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    return search


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metric_columns = ["accuracy", "precision", "recall", "f1"]
    metrics.set_index("model")[metric_columns].T.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_ylim(0.9, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Сравнение качества моделей")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].bar(metrics["model"], metrics["train_time_seconds"], color=["#4c78a8", "#f58518"])
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Время обучения")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrices(matrices: dict[str, np.ndarray], target_names: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(9, 4))
    if len(matrices) == 1:
        axes = [axes]

    for ax, (name, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(target_names)), labels=target_names, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(target_names)), labels=target_names)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    top = feature_importance.head(top_n).iloc[::-1]
    y_pos = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        y_pos - 0.18,
        top["importance_mean"],
        height=0.36,
        xerr=top["importance_std"],
        label="OOB^j permutation",
        color="#4c78a8",
        alpha=0.9,
    )
    ax.barh(
        y_pos + 0.18,
        top["gini_importance"],
        height=0.36,
        label="Gini importance",
        color="#f58518",
        alpha=0.8,
    )
    ax.set_yticks(y_pos, labels=top["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Топ-{top_n} признаков по OOB^j")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    data = make_splits(random_state=RANDOM_STATE)

    grid_search = run_oob_grid_search(data.X_train, data.y_train)
    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_columns = [
        "param_n_estimators",
        "param_max_features",
        "param_max_depth",
        "param_min_samples_leaf",
        "mean_test_score",
        "rank_test_score",
        "mean_fit_time",
    ]
    grid_results[grid_columns].sort_values("rank_test_score").to_csv(
        ARTIFACTS_DIR / "grid_results.csv",
        index=False,
    )

    best_params = dict(grid_search.best_params_)

    custom = OOBRandomForestClassifier(
        **best_params,
        bootstrap=True,
        random_state=RANDOM_STATE,
    )
    start = time.perf_counter()
    custom.fit(data.X_train, data.y_train)
    custom_train_time = time.perf_counter() - start
    custom_pred = custom.predict(data.X_test)

    sklearn_forest = SklearnRandomForestClassifier(
        **best_params,
        bootstrap=True,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    start = time.perf_counter()
    sklearn_forest.fit(data.X_train, data.y_train)
    sklearn_train_time = time.perf_counter() - start
    sklearn_pred = sklearn_forest.predict(data.X_test)

    metrics = pd.DataFrame(
        [
            evaluate_model("custom OOB Random Forest", data.y_test, custom_pred, custom_train_time),
            evaluate_model("sklearn RandomForestClassifier", data.y_test, sklearn_pred, sklearn_train_time),
        ]
    )
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)

    oob_importance = custom.compute_oob_permutation_importance(
        data.X_train,
        data.y_train,
        n_repeats=7,
        random_state=RANDOM_STATE,
    )
    gini_importance = pd.DataFrame(
        {
            "feature": data.feature_names,
            "gini_importance": custom.feature_importances_,
        }
    )
    feature_importance = oob_importance.merge(gini_importance, on="feature", how="left")
    feature_importance.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)

    confusion_matrices = {
        "Custom RF": confusion_matrix(data.y_test, custom_pred),
        "Sklearn RF": confusion_matrix(data.y_test, sklearn_pred),
    }
    plot_metrics(metrics, IMAGES_DIR / "metrics_comparison.png")
    plot_confusion_matrices(confusion_matrices, data.target_names, IMAGES_DIR / "confusion_matrices.png")
    plot_feature_importance(feature_importance, IMAGES_DIR / "feature_importance.png")

    summary = {
        "dataset": data.source_name,
        "target_names": data.target_names,
        "sizes": {
            "train": len(data.X_train),
            "test": len(data.X_test),
            "features": len(data.feature_names),
        },
        "best_params": best_params,
        "best_oob_accuracy_from_grid": float(grid_search.best_score_),
        "custom_oob_accuracy": float(custom.oob_score_),
        "sklearn_oob_accuracy": float(sklearn_forest.oob_score_),
        "confusion_matrices": {
            "custom": confusion_matrices["Custom RF"].tolist(),
            "sklearn": confusion_matrices["Sklearn RF"].tolist(),
        },
        "images": {
            "metrics_comparison": "images/metrics_comparison.png",
            "confusion_matrices": "images/confusion_matrices.png",
            "feature_importance": "images/feature_importance.png",
        },
        "top_oob_features": feature_importance.head(10).to_dict(orient="records"),
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Dataset: {data.source_name}")
    print(f"Train/test sizes: {len(data.X_train)}/{len(data.X_test)}")
    print(f"Best params by OOB GridSearchCV: {best_params}")
    print(f"Best OOB accuracy: {grid_search.best_score_:.4f}")
    print("\nMetrics:")
    print(metrics.round(4).to_string(index=False))
    print("\nTop OOB^j feature importances:")
    print(feature_importance.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
