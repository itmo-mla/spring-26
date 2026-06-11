from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from data import make_splits
from gmm import GaussianMixtureEM


LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"
IMAGES_DIR = LAB_DIR / "images"
RANDOM_STATE = 42
MODEL_PARAMS = {
    "n_components": 3,
    "tol": 1e-4,
    "reg_covar": 1e-4,
    "max_iter": 300,
    "n_init": 5,
    "random_state": RANDOM_STATE,
}


def clustering_accuracy(y_true: np.ndarray, labels: np.ndarray) -> float:
    """Return best label-matching accuracy for small cluster counts."""
    true_values = np.unique(y_true)
    label_values = np.unique(labels)
    if len(true_values) != len(label_values) or len(label_values) > 8:
        return float("nan")

    best = 0
    for permutation in itertools.permutations(true_values):
        mapping = dict(zip(label_values, permutation))
        remapped = np.array([mapping[label] for label in labels])
        best = max(best, int(np.sum(remapped == y_true)))
    return best / len(y_true)


def evaluate_model(
    name: str,
    model: GaussianMixtureEM | GaussianMixture,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    train_time: float,
) -> dict[str, Any]:
    test_labels = model.predict(X_test)
    train_log_likelihood = float(np.sum(model.score_samples(X_train)))
    test_log_likelihood = float(np.sum(model.score_samples(X_test)))

    return {
        "model": name,
        "train_log_likelihood": train_log_likelihood,
        "test_log_likelihood": test_log_likelihood,
        "train_avg_log_likelihood": float(model.score(X_train)),
        "test_avg_log_likelihood": float(model.score(X_test)),
        "test_bic": float(model.bic(X_test)),
        "test_aic": float(model.aic(X_test)),
        "ari_vs_true_labels": adjusted_rand_score(y_test, test_labels),
        "nmi_vs_true_labels": normalized_mutual_info_score(y_test, test_labels),
        "cluster_accuracy_vs_true_labels": clustering_accuracy(y_test, test_labels),
        "n_iter": int(model.n_iter_),
        "converged": bool(model.converged_),
        "train_time_seconds": train_time,
    }


def make_component_summary(
    model_name: str,
    model: GaussianMixtureEM | GaussianMixture,
    feature_names: list[str],
    scaler: Any,
) -> pd.DataFrame:
    original_means = scaler.inverse_transform(model.means_)
    rows = []
    for component, (weight, means) in enumerate(zip(model.weights_, original_means)):
        row = {
            "model": model_name,
            "component": component,
            "weight": weight,
        }
        row.update({feature: value for feature, value in zip(feature_names, means)})
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(
    metrics: pd.DataFrame,
    log_likelihood_curve: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    custom_labels: np.ndarray,
    sklearn_labels: np.ndarray,
) -> dict[str, str]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(
        log_likelihood_curve["iteration"],
        log_likelihood_curve["custom_train_avg_log_likelihood"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Итерация")
    plt.ylabel("Средний log-likelihood")
    plt.title("Сходимость EM-алгоритма")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    convergence_path = IMAGES_DIR / "em_convergence.png"
    plt.savefig(convergence_path, dpi=160)
    plt.savefig(IMAGES_DIR / "em_convergence.svg")
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width / 2, metrics["train_avg_log_likelihood"], width, label="Train")
    plt.bar(x + width / 2, metrics["test_avg_log_likelihood"], width, label="Test")
    plt.xticks(x, metrics["model"], rotation=10, ha="right")
    plt.ylabel("Средний log-likelihood")
    plt.title("Сравнение правдоподобия GMM")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    likelihood_path = IMAGES_DIR / "likelihood_comparison.png"
    plt.savefig(likelihood_path, dpi=160)
    plt.savefig(IMAGES_DIR / "likelihood_comparison.svg")
    plt.close()

    projected = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_test)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    plots = [
        ("Истинные классы", y_test),
        ("Custom GaussianMixtureEM", custom_labels),
        ("sklearn GaussianMixture", sklearn_labels),
    ]
    for ax, (title, labels) in zip(axes, plots):
        scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="viridis", s=45, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("PCA 2")
    fig.colorbar(scatter, ax=axes, shrink=0.85, label="Класс / компонента")
    fig.suptitle("Компоненты GMM в двумерной PCA-проекции", y=1.03)
    fig.tight_layout()
    clusters_path = IMAGES_DIR / "clusters_pca.png"
    fig.savefig(clusters_path, dpi=160, bbox_inches="tight")
    fig.savefig(IMAGES_DIR / "clusters_pca.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    matrices = [
        ("Custom GaussianMixtureEM", pd.crosstab(y_test, custom_labels)),
        ("sklearn GaussianMixture", pd.crosstab(y_test, sklearn_labels)),
    ]
    for ax, (title, matrix) in zip(axes, matrices):
        im = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Компонента")
        ax.set_xticks(np.arange(matrix.shape[1]), labels=matrix.columns)
        ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                text_color = "white" if value > matrix.to_numpy().max() * 0.5 else "black"
                ax.text(column, row, value, ha="center", va="center", color=text_color, fontweight="bold")
    axes[0].set_ylabel("Истинный класс")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Количество объектов")
    fig.suptitle("Соответствие истинных классов и компонент GMM", y=1.03)
    fig.tight_layout()
    component_matrix_path = IMAGES_DIR / "component_matrix.png"
    fig.savefig(component_matrix_path, dpi=160, bbox_inches="tight")
    fig.savefig(IMAGES_DIR / "component_matrix.svg", bbox_inches="tight")
    plt.close(fig)

    return {
        "convergence": str(convergence_path.relative_to(LAB_DIR)),
        "likelihood_comparison": str(likelihood_path.relative_to(LAB_DIR)),
        "clusters_pca": str(clusters_path.relative_to(LAB_DIR)),
        "component_matrix": str(component_matrix_path.relative_to(LAB_DIR)),
    }


def run_experiment() -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data = make_splits(random_state=RANDOM_STATE)

    custom = GaussianMixtureEM(**MODEL_PARAMS)
    start = time.perf_counter()
    custom.fit(data.X_train)
    custom_train_time = time.perf_counter() - start

    sklearn_model = GaussianMixture(
        n_components=MODEL_PARAMS["n_components"],
        covariance_type="full",
        tol=MODEL_PARAMS["tol"],
        reg_covar=MODEL_PARAMS["reg_covar"],
        max_iter=MODEL_PARAMS["max_iter"],
        n_init=MODEL_PARAMS["n_init"],
        random_state=MODEL_PARAMS["random_state"],
    )
    start = time.perf_counter()
    sklearn_model.fit(data.X_train)
    sklearn_train_time = time.perf_counter() - start
    custom_test_labels = custom.predict(data.X_test)
    sklearn_test_labels = sklearn_model.predict(data.X_test)

    metrics = pd.DataFrame(
        [
            evaluate_model("Custom GaussianMixtureEM", custom, data.X_train, data.X_test, data.y_test, custom_train_time),
            evaluate_model(
                "sklearn GaussianMixture",
                sklearn_model,
                data.X_train,
                data.X_test,
                data.y_test,
                sklearn_train_time,
            ),
        ]
    )
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)

    predictions = pd.DataFrame(
        {
            "y_true": data.y_test,
            "custom_component": custom_test_labels,
            "sklearn_component": sklearn_test_labels,
            "custom_log_density": custom.score_samples(data.X_test),
            "sklearn_log_density": sklearn_model.score_samples(data.X_test),
        }
    )
    predictions.to_csv(ARTIFACTS_DIR / "predictions.csv", index=False)

    component_summary = pd.concat(
        [
            make_component_summary("Custom GaussianMixtureEM", custom, data.feature_names, data.scaler),
            make_component_summary("sklearn GaussianMixture", sklearn_model, data.feature_names, data.scaler),
        ],
        ignore_index=True,
    )
    component_summary.to_csv(ARTIFACTS_DIR / "component_summary.csv", index=False)

    log_likelihood_curve = pd.DataFrame(
        {
            "iteration": np.arange(1, custom.n_iter_ + 1),
            "custom_train_avg_log_likelihood": custom.lower_bound_history_,
        }
    )
    log_likelihood_curve.to_csv(ARTIFACTS_DIR / "log_likelihood_curve.csv", index=False)
    images = save_plots(
        metrics,
        log_likelihood_curve,
        data.X_test,
        data.y_test,
        custom_test_labels,
        sklearn_test_labels,
    )

    summary = {
        "dataset": data.source_name,
        "sizes": {
            "train": len(data.X_train),
            "test": len(data.X_test),
            "features": len(data.feature_names),
            "classes": len(data.target_names),
        },
        "target_names": data.target_names,
        "feature_names": data.feature_names,
        "model_params": MODEL_PARAMS,
        "metrics": metrics.to_dict(orient="records"),
        "custom_weights": custom.weights_.tolist(),
        "sklearn_weights": sklearn_model.weights_.tolist(),
        "custom_convergence": {
            "n_iter": custom.n_iter_,
            "converged": custom.converged_,
            "initial_train_avg_log_likelihood": custom.lower_bound_history_[0],
            "final_train_avg_log_likelihood": custom.lower_bound_history_[-1],
        },
        "images": images,
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run_experiment()
    metrics = pd.DataFrame(summary["metrics"])

    print(f"Dataset: {summary['dataset']}")
    print(f"Train/test sizes: {summary['sizes']['train']}/{summary['sizes']['test']}")
    print(f"Model params: {summary['model_params']}")
    print("\nMetrics:")
    print(metrics.round(4).to_string(index=False))
    print("\nCustom EM convergence:")
    print(summary["custom_convergence"])


if __name__ == "__main__":
    main()
