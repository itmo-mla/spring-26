from itertools import permutations
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gmm import GaussianMixtureEM


RANDOM_STATE = 42
N_COMPONENTS = 3
TEST_SIZE = 0.2
K_RANGE = range(1, 8)
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataset = load_iris(as_frame=True)
    x = dataset.data
    y = dataset.target.to_numpy(dtype=int)
    target_names = dataset.target_names
    return x, y, target_names


def split_and_scale(
    x: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)
    return x_train, x_test, y_train, y_test


def build_custom_model(
    n_components: int = N_COMPONENTS,
    init_params: str = "kmeans++",
    n_init: int = 10,
) -> GaussianMixtureEM:
    return GaussianMixtureEM(
        n_components=n_components,
        max_iter=300,
        tol=1e-5,
        reg_covar=1e-6,
        n_init=n_init,
        init_params=init_params,
        random_state=RANDOM_STATE,
    )


def build_sklearn_model(
    n_components: int = N_COMPONENTS,
    init_params: str = "kmeans",
    n_init: int = 10,
) -> GaussianMixture:
    return GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=300,
        tol=1e-5,
        reg_covar=1e-6,
        n_init=n_init,
        init_params=init_params,
        random_state=RANDOM_STATE,
    )


def clustering_accuracy(
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    n_components: int,
) -> tuple[float, dict[int, int], np.ndarray]:
    classes = np.unique(y_true)
    clusters = np.arange(n_components)
    if len(classes) != len(clusters):
        raise ValueError("number of classes must match number of clusters for accuracy calculation")

    best_accuracy = -1.0
    best_mapping: dict[int, int] = {}
    best_predictions = np.empty_like(y_true)

    for class_order in permutations(classes):
        mapping = {int(cluster): int(class_label) for cluster, class_label in zip(clusters, class_order, strict=True)}
        predictions = np.array([mapping[int(label)] for label in cluster_labels], dtype=int)
        accuracy = float(np.mean(predictions == y_true))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping
            best_predictions = predictions

    return best_accuracy, best_mapping, best_predictions


def evaluate_model(
    model,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    clusters = model.predict(x_test)
    accuracy, mapping, mapped_predictions = clustering_accuracy(y_test, clusters, model.n_components)
    return {
        "train_ll": float(model.score(x_train)),
        "test_ll": float(model.score(x_test)),
        "bic_train": float(model.bic(x_train)),
        "aic_train": float(model.aic(x_train)),
        "accuracy_test": accuracy,
        "ari_test": float(adjusted_rand_score(y_test, clusters)),
        "mapping": mapping,
        "confusion_matrix": confusion_matrix(y_test, mapped_predictions),
        "clusters": clusters,
        "mapped_predictions": mapped_predictions,
        "weights": np.asarray(model.weights_).copy(),
    }


def fit_and_evaluate(
    model_name: str,
    factory,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float | int | str | bool], dict[str, object]]:
    model = factory()
    start_time = perf_counter()
    model.fit(x_train)
    fit_time = perf_counter() - start_time

    metrics = evaluate_model(model, x_train, x_test, y_test)
    row = {
        "model": model_name,
        "train_ll": metrics["train_ll"],
        "test_ll": metrics["test_ll"],
        "bic_train": metrics["bic_train"],
        "aic_train": metrics["aic_train"],
        "accuracy_test": metrics["accuracy_test"],
        "ari_test": metrics["ari_test"],
        "fit_time_sec": fit_time,
        "n_iter": int(model.n_iter_),
        "converged": bool(model.converged_),
    }
    fitted = {
        "model": model,
        "fit_time_sec": fit_time,
        **metrics,
    }
    return row, fitted


def run_main_experiment(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | int | str | bool]] = []
    fitted_models: dict[str, object] = {}

    for model_name, factory in (
        ("Собственная GMM", lambda: build_custom_model(init_params="kmeans++")),
        ("sklearn GMM", lambda: build_sklearn_model(init_params="kmeans")),
    ):
        row, fitted = fit_and_evaluate(model_name, factory, x_train, x_test, y_test)
        rows.append(row)
        fitted_models[model_name] = fitted

    results = pd.DataFrame(rows)
    custom_ll = float(results.loc[results["model"] == "Собственная GMM", "test_ll"].iloc[0])
    sklearn_ll = float(results.loc[results["model"] == "sklearn GMM", "test_ll"].iloc[0])
    results["delta_test_ll"] = np.nan
    results.loc[results["model"] == "Собственная GMM", "delta_test_ll"] = abs(custom_ll - sklearn_ll)
    return results, fitted_models


def evaluate_density(model, x_train: np.ndarray, x_test: np.ndarray) -> dict[str, float]:
    return {
        "train_ll": float(model.score(x_train)),
        "test_ll": float(model.score(x_test)),
        "bic_train": float(model.bic(x_train)),
        "aic_train": float(model.aic(x_train)),
    }


def fit_model_density(
    model_name: str,
    factory,
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[dict[str, float | int | str | bool], object]:
    model = factory()
    start_time = perf_counter()
    model.fit(x_train)
    fit_time = perf_counter() - start_time
    metrics = evaluate_density(model, x_train, x_test)
    row = {
        "model": model_name,
        **metrics,
        "fit_time_sec": fit_time,
        "n_iter": int(model.n_iter_),
        "converged": bool(model.converged_),
    }
    return row, model


def run_k_sweep(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []

    for k in K_RANGE:
        for model_name, factory in (
            ("Собственная GMM", lambda k=k: build_custom_model(n_components=k, init_params="kmeans++")),
            ("sklearn GMM", lambda k=k: build_sklearn_model(n_components=k, init_params="kmeans")),
        ):
            row, _ = fit_model_density(model_name, factory, x_train, x_test)
            rows.append({"k": k, **row})

    return pd.DataFrame(rows)


def run_init_experiment(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    configs = (
        ("Custom kmeans++", lambda: build_custom_model(init_params="kmeans++")),
        ("sklearn kmeans", lambda: build_sklearn_model(init_params="kmeans")),
        ("Custom random", lambda: build_custom_model(init_params="random")),
        ("sklearn random", lambda: build_sklearn_model(init_params="random")),
    )

    rows: list[dict[str, float | int | str | bool]] = []
    fitted: dict[str, object] = {}

    for model_name, factory in configs:
        row, fitted_result = fit_and_evaluate(model_name, factory, x_train, x_test, y_test)
        rows.append(row)
        fitted[model_name] = fitted_result

    return pd.DataFrame(rows), fitted


def plot_metric_comparison(results: pd.DataFrame) -> None:
    metrics = [
        ("test_ll", "Test Log-Likelihood"),
        ("accuracy_test", "Test Accuracy"),
        ("ari_test", "Test ARI"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metric_axes = axes.flat[:3]

    for ax, (metric, title) in zip(metric_axes, metrics, strict=True):
        bars = ax.bar(results["model"], results[metric], color=["#4C72B0", "#55A868"])
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=10)

    delta_ll = float(results["delta_test_ll"].dropna().iloc[0])
    delta_ax = axes[1, 1]
    delta_ax.bar(["|Δ test LL|"], [max(delta_ll, 1e-16)], color="#C44E52")
    delta_ax.set_yscale("log")
    delta_ax.set_title("Разность test log-likelihood")
    delta_ax.bar_label(delta_ax.containers[0], labels=[f"{delta_ll:.3e}"], padding=3)
    delta_ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Сравнение собственной GMM и sklearn (K=3, k-means инициализация)")
    plt.savefig(ARTIFACTS_DIR / "metrics_comparison.png", dpi=160)
    plt.close()


def plot_likelihood_curve(
    custom_model: GaussianMixtureEM,
    sklearn_model: GaussianMixture,
    x_train: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    custom_history = custom_model.log_likelihood_history_
    ax.plot(
        range(1, len(custom_history) + 1),
        custom_history,
        marker="o",
        label="Собственная GMM",
    )

    sklearn_final_ll = float(sklearn_model.score(x_train))
    ax.axhline(
        sklearn_final_ll,
        color="#55A868",
        linestyle="--",
        label=f"sklearn final LL = {sklearn_final_ll:.4f}",
    )

    ax.set_title("Сходимость EM-алгоритма")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Средний log-likelihood (train)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "likelihood_curve.png", dpi=160)
    plt.close()


def plot_pca_clusters(
    x_test: np.ndarray,
    y_test: np.ndarray,
    fitted_models: dict[str, object],
    target_names: np.ndarray,
) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projected = pca.fit_transform(x_test)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels: list[tuple[str, np.ndarray]] = [("Истинные классы", y_test)]

    for model_name, result in fitted_models.items():
        title = (
            f"{model_name}\n"
            f"Acc={result['accuracy_test']:.3f}, ARI={result['ari_test']:.3f}"
        )
        panels.append((title, result["mapped_predictions"]))

    for ax, (title, labels) in zip(axes, panels, strict=True):
        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=labels,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=[str(name) for name in target_names],
            title="Класс",
            loc="best",
        )

    fig.suptitle("PCA-проекция тестовой выборки")
    plt.savefig(ARTIFACTS_DIR / "pca_clusters.png", dpi=160)
    plt.close()


def plot_bic_aic_vs_k(k_sweep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for ax, metric, ylabel in zip(
        axes,
        ("bic_train", "aic_train"),
        ("BIC (train)", "AIC (train)"),
        strict=True,
    ):
        for model_name, color in (
            ("Собственная GMM", "#4C72B0"),
            ("sklearn GMM", "#55A868"),
        ):
            subset = k_sweep[k_sweep["model"] == model_name]
            ax.plot(subset["k"], subset[metric], marker="o", label=model_name, color=color)

        custom_subset = k_sweep[k_sweep["model"] == "Собственная GMM"]
        best_row = custom_subset.loc[custom_subset[metric].idxmin()]
        ax.axvline(int(best_row["k"]), color="#C44E52", linestyle="--", alpha=0.7)
        ax.set_title(f"{ylabel}: минимум custom при K={int(best_row['k'])} ({best_row[metric]:.1f})")
        ax.set_xlabel("Число компонент K")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("Выбор числа компонент по информационным критериям")
    plt.savefig(ARTIFACTS_DIR / "bic_aic_vs_k.png", dpi=160)
    plt.close()


def plot_init_comparison(
    x_test: np.ndarray,
    y_test: np.ndarray,
    init_results: pd.DataFrame,
    init_fitted: dict[str, object],
    target_names: np.ndarray,
) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projected = pca.fit_transform(x_test)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panel_order = ["Custom kmeans++", "sklearn kmeans", "Custom random", "sklearn random"]

    for ax, model_name in zip(axes.flat, panel_order, strict=True):
        result = init_fitted[model_name]
        row = init_results[init_results["model"] == model_name].iloc[0]
        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=result["mapped_predictions"],
            cmap="viridis",
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(
            f"{model_name}\n"
            f"test LL={row['test_ll']:.4f}, Acc={row['accuracy_test']:.3f}, ARI={row['ari_test']:.3f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=[str(name) for name in target_names],
            title="Класс",
            loc="best",
            fontsize=8,
        )

    fig.suptitle("Влияние инициализации на качество GMM (K=3)")
    plt.savefig(ARTIFACTS_DIR / "init_comparison.png", dpi=160)
    plt.close()


def save_run_summary(
    x: pd.DataFrame,
    y: np.ndarray,
    target_names: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    results: pd.DataFrame,
    fitted_models: dict[str, object],
    k_sweep: pd.DataFrame,
    init_results: pd.DataFrame,
) -> None:
    custom_row = results[results["model"] == "Собственная GMM"].iloc[0]
    sklearn_row = results[results["model"] == "sklearn GMM"].iloc[0]
    custom_weights = fitted_models["Собственная GMM"]["weights"]
    sklearn_weights = fitted_models["sklearn GMM"]["weights"]

    lines = [
        "# Сводка запуска lab4",
        "",
        f"Датасет: Iris из `sklearn.datasets`, объектов: {x.shape[0]}, признаков: {x.shape[1]}.",
        "Классы: " + ", ".join(f"{name}={int(np.sum(y == index))}" for index, name in enumerate(target_names)) + ".",
        f"Train/test split: {x_train.shape[0]}/{x_test.shape[0]} (80/20, stratified).",
        "",
        "## Основное сравнение (K=3, k-means инициализация)",
        "",
        "| Модель | Train LL | Test LL | BIC | AIC | Acc (test) | ARI (test) | Fit time, sec | Iterations |",
        "|--------|----------|---------|-----|-----|------------|------------|---------------|------------|",
    ]

    for _, row in results.iterrows():
        lines.append(
            "| {model} | {train_ll:.4f} | {test_ll:.4f} | {bic:.2f} | {aic:.2f} | "
            "{accuracy:.4f} | {ari:.4f} | {time:.3f} | {n_iter} |".format(
                model=row["model"],
                train_ll=row["train_ll"],
                test_ll=row["test_ll"],
                bic=row["bic_train"],
                aic=row["aic_train"],
                accuracy=row["accuracy_test"],
                ari=row["ari_test"],
                time=row["fit_time_sec"],
                n_iter=int(row["n_iter"]),
            )
        )

    delta_ll = abs(float(custom_row["test_ll"]) - float(sklearn_row["test_ll"]))
    lines.extend(
        [
            "",
            f"|Δ test LL| = {delta_ll:.3e}",
            "",
            "### Веса компонент",
            "",
            "| Компонента | Custom | sklearn | |Δ| |",
            "|------------|--------|---------|-----|",
        ]
    )
    for index, (custom_weight, sklearn_weight) in enumerate(zip(custom_weights, sklearn_weights, strict=True)):
        lines.append(
            f"| {index} | {custom_weight:.6f} | {sklearn_weight:.6f} | {abs(custom_weight - sklearn_weight):.3e} |"
        )

    lines.extend(["", "## Перебор K (BIC/AIC на train)", ""])
    lines.append("| K | Custom BIC | sklearn BIC | Custom AIC | sklearn AIC | Custom test LL | sklearn test LL |")
    lines.append("|---|------------|-------------|------------|-------------|----------------|-----------------|")
    for k in K_RANGE:
        custom_k = k_sweep[(k_sweep["k"] == k) & (k_sweep["model"] == "Собственная GMM")].iloc[0]
        sklearn_k = k_sweep[(k_sweep["k"] == k) & (k_sweep["model"] == "sklearn GMM")].iloc[0]
        lines.append(
            f"| {k} | {custom_k['bic_train']:.2f} | {sklearn_k['bic_train']:.2f} | "
            f"{custom_k['aic_train']:.2f} | {sklearn_k['aic_train']:.2f} | "
            f"{custom_k['test_ll']:.4f} | {sklearn_k['test_ll']:.4f} |"
        )

    lines.extend(["", "## Эксперимент с инициализацией (K=3)", ""])
    lines.append("| Модель | Test LL | Acc (test) | ARI (test) | Iterations |")
    lines.append("|--------|---------|------------|------------|------------|")
    for _, row in init_results.iterrows():
        lines.append(
            f"| {row['model']} | {row['test_ll']:.4f} | {row['accuracy_test']:.4f} | "
            f"{row['ari_test']:.4f} | {int(row['n_iter'])} |"
        )

    lines.extend(["", "## Матрицы ошибок на test после оптимального сопоставления компонент", ""])
    for model_name, result in fitted_models.items():
        lines.append(f"### {model_name}")
        lines.append("")
        matrix = result["confusion_matrix"]
        lines.extend(
            [
                "| Истинный \\ Предсказанный | class_0 | class_1 | class_2 |",
                "|--------------------------|---------|---------|---------|",
            ]
        )
        for row_index, row in enumerate(matrix):
            lines.append(f"| class_{row_index} | {row[0]} | {row[1]} | {row[2]} |")
        lines.append("")

    (ARTIFACTS_DIR / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    x, y, target_names = load_dataset()
    x_train, x_test, y_train, y_test = split_and_scale(x, y)

    results, fitted_models = run_main_experiment(x_train, x_test, y_test)
    k_sweep = run_k_sweep(x_train, x_test)
    init_results, init_fitted = run_init_experiment(x_train, x_test, y_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ARTIFACTS_DIR / "results.csv", index=False)
    k_sweep.to_csv(ARTIFACTS_DIR / "k_sweep.csv", index=False)
    init_results.to_csv(ARTIFACTS_DIR / "init_comparison.csv", index=False)

    custom_model = fitted_models["Собственная GMM"]["model"]
    sklearn_model = fitted_models["sklearn GMM"]["model"]
    plot_metric_comparison(results)
    plot_likelihood_curve(custom_model, sklearn_model, x_train)
    plot_pca_clusters(x_test, y_test, fitted_models, target_names)
    plot_bic_aic_vs_k(k_sweep)
    plot_init_comparison(x_test, y_test, init_results, init_fitted, target_names)

    save_run_summary(
        x,
        y,
        target_names,
        x_train,
        x_test,
        y_test,
        results,
        fitted_models,
        k_sweep,
        init_results,
    )
    print((ARTIFACTS_DIR / "run_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
