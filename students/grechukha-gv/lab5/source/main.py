from pathlib import Path
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet

from lsa import LatentSemanticAnalysis
from metrics import ndcg_at_k, rmse_on_entries
from slim import SlimRecommender


RANDOM_STATE = 42
MAX_FEATURES = 1000
TEST_SIZE = 0.2
N_COMPONENTS = 45
NDCG_K = 10
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CATEGORIES = (
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.misc",
)


def load_dataset() -> tuple[sparse.csr_matrix, np.ndarray, list[str], list[str]]:
    dataset = fetch_20newsgroups(
        subset="train",
        categories=list(CATEGORIES),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    texts = dataset.data
    targets = dataset.target

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=3,
        max_df=0.85,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts).tocsr()
    return matrix, targets, list(dataset.target_names), list(vectorizer.get_feature_names_out())


def make_holdout(
    matrix: sparse.csr_matrix,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train = matrix.tolil(copy=True)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_columns = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if row_columns.size < 2:
            continue

        n_holdout = min(max(1, int(np.floor(row_columns.size * test_size))), row_columns.size - 1)
        selected = rng.choice(row_columns.size, size=n_holdout, replace=False)
        for position in selected:
            column = int(row_columns[position])
            rows.append(row)
            columns.append(column)
            values.append(float(row_values[position]))
            train[row, column] = 0.0

    train = train.tocsr()
    train.eliminate_zeros()
    return train, np.array(rows), np.array(columns), np.array(values)


def build_custom_slim() -> SlimRecommender:
    return SlimRecommender(alpha_l1=0.0008, alpha_l2=0.02, max_iter=260, tol=1e-6, random_state=RANDOM_STATE)


def fit_reference_slim(train_matrix: sparse.csr_matrix) -> tuple[np.ndarray, int]:
    x_dense = train_matrix.toarray()
    n_items = x_dense.shape[1]
    coefficients = np.zeros((n_items, n_items), dtype=float)
    total_iterations = 0

    alpha_l1 = 0.0008
    alpha_l2 = 0.02
    alpha = alpha_l1 + alpha_l2
    l1_ratio = alpha_l1 / alpha

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for item in range(n_items):
            feature_mask = np.ones(n_items, dtype=bool)
            feature_mask[item] = False
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                positive=True,
                fit_intercept=False,
                max_iter=2500,
                tol=1e-4,
                selection="random",
                random_state=RANDOM_STATE,
            )
            model.fit(x_dense[:, feature_mask], x_dense[:, item])
            coefficients[feature_mask, item] = model.coef_
            total_iterations += int(model.n_iter_)

    return coefficients, total_iterations


def evaluate_predictions(
    model_name: str,
    predictions: np.ndarray,
    train_matrix: sparse.csr_matrix,
    holdout_rows: np.ndarray,
    holdout_columns: np.ndarray,
    holdout_values: np.ndarray,
    fit_time: float,
    n_iter: int,
) -> dict[str, float | int | str]:
    predictions = np.maximum(predictions, 0.0)
    return {
        "model": model_name,
        "rmse": rmse_on_entries(predictions, holdout_rows, holdout_columns, holdout_values),
        f"ndcg@{NDCG_K}": ndcg_at_k(
            predictions,
            train_matrix,
            holdout_rows,
            holdout_columns,
            holdout_values,
            k=NDCG_K,
        ),
        "fit_time_sec": fit_time,
        "n_iter": n_iter,
    }


def run_experiment(
    train_matrix: sparse.csr_matrix,
    holdout_rows: np.ndarray,
    holdout_columns: np.ndarray,
    holdout_values: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | int | str]] = []
    fitted: dict[str, object] = {}

    custom_slim = build_custom_slim()
    start_time = perf_counter()
    custom_slim.fit(train_matrix)
    fit_time = perf_counter() - start_time
    predictions = custom_slim.predict(train_matrix)
    rows.append(
        evaluate_predictions(
            "Собственный SLIM",
            predictions,
            train_matrix,
            holdout_rows,
            holdout_columns,
            holdout_values,
            fit_time,
            custom_slim.n_iter_,
        )
    )
    fitted["Собственный SLIM"] = custom_slim

    start_time = perf_counter()
    reference_slim_coef, reference_slim_iterations = fit_reference_slim(train_matrix)
    fit_time = perf_counter() - start_time
    predictions = train_matrix @ reference_slim_coef
    rows.append(
        evaluate_predictions(
            "sklearn ElasticNet SLIM",
            np.asarray(predictions),
            train_matrix,
            holdout_rows,
            holdout_columns,
            holdout_values,
            fit_time,
            reference_slim_iterations,
        )
    )
    fitted["sklearn ElasticNet SLIM"] = reference_slim_coef

    custom_lsa = LatentSemanticAnalysis(n_components=N_COMPONENTS)
    start_time = perf_counter()
    custom_lsa.fit(train_matrix)
    fit_time = perf_counter() - start_time
    predictions = custom_lsa.reconstruct(train_matrix)
    rows.append(
        evaluate_predictions(
            "Собственная LSA",
            predictions,
            train_matrix,
            holdout_rows,
            holdout_columns,
            holdout_values,
            fit_time,
            N_COMPONENTS,
        )
    )
    fitted["Собственная LSA"] = custom_lsa

    reference_lsa = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE, n_iter=10)
    start_time = perf_counter()
    transformed = reference_lsa.fit_transform(train_matrix)
    fit_time = perf_counter() - start_time
    predictions = reference_lsa.inverse_transform(transformed)
    rows.append(
        evaluate_predictions(
            "sklearn TruncatedSVD",
            predictions,
            train_matrix,
            holdout_rows,
            holdout_columns,
            holdout_values,
            fit_time,
            N_COMPONENTS,
        )
    )
    fitted["sklearn TruncatedSVD"] = reference_lsa

    return pd.DataFrame(rows), fitted


def plot_metric_comparison(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    axes[0].bar(results["model"], results["rmse"], color=colors)
    axes[0].set_title("RMSE на скрытых TF-IDF значениях")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(results["model"], results[f"ndcg@{NDCG_K}"], color=colors)
    axes[1].set_title(f"NDCG@{NDCG_K} для рекомендаций терминов")
    axes[1].set_ylabel("NDCG")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Сравнение собственных моделей с эталонными реализациями")
    plt.savefig(ARTIFACTS_DIR / "metrics_comparison.png", dpi=160)
    plt.close()


def plot_fit_time(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results["model"], results["fit_time_sec"], color="#4C72B0")
    ax.set_title("Время обучения")
    ax.set_ylabel("Секунды")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "fit_time.png", dpi=160)
    plt.close()


def plot_slim_coefficients(fitted: dict[str, object]) -> None:
    custom = fitted["Собственный SLIM"].coef_
    reference = fitted["sklearn ElasticNet SLIM"]
    if custom is None:
        return

    labels = ["Собственный SLIM", "sklearn ElasticNet SLIM"]
    densities = [np.mean(custom > 1e-12), np.mean(reference > 1e-12)]
    mean_weights = [
        float(np.mean(custom[custom > 1e-12])) if np.any(custom > 1e-12) else 0.0,
        float(np.mean(reference[reference > 1e-12])) if np.any(reference > 1e-12) else 0.0,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].bar(labels, densities, color=["#4C72B0", "#55A868"])
    axes[0].set_title("Доля ненулевых весов")
    axes[0].set_ylabel("Density")
    axes[0].tick_params(axis="x", rotation=10)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, mean_weights, color=["#4C72B0", "#55A868"])
    axes[1].set_title("Средний ненулевой вес")
    axes[1].tick_params(axis="x", rotation=10)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Разреженность матриц похожести SLIM")
    plt.savefig(ARTIFACTS_DIR / "slim_coefficients.png", dpi=160)
    plt.close()


def plot_slim_convergence(fitted: dict[str, object]) -> None:
    slim = fitted["Собственный SLIM"]
    history = getattr(slim, "loss_history_", None)
    if not history:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o", color="#4C72B0")
    ax.set_title("Сходимость собственного SLIM")
    ax.set_xlabel("Итерация proximal gradient")
    ax.set_ylabel("Значение целевой функции")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "slim_convergence.png", dpi=160)
    plt.close()


def plot_lsa_singular_values(fitted: dict[str, object]) -> None:
    custom = fitted["Собственная LSA"].singular_values_
    reference = fitted["sklearn TruncatedSVD"].singular_values_
    if custom is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(custom) + 1), custom, marker="o", label="Собственная LSA")
    ax.plot(range(1, len(reference) + 1), reference, marker="x", label="sklearn TruncatedSVD")
    ax.set_title("Сингулярные значения LSA")
    ax.set_xlabel("Компонента")
    ax.set_ylabel("Сингулярное значение")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "lsa_singular_values.png", dpi=160)
    plt.close()


def save_results(results: pd.DataFrame, fitted: dict[str, object]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ARTIFACTS_DIR / "results.csv", index=False)
    plot_metric_comparison(results)
    plot_fit_time(results)
    plot_slim_coefficients(fitted)
    plot_slim_convergence(fitted)
    plot_lsa_singular_values(fitted)


def save_run_summary(
    matrix: sparse.csr_matrix,
    targets: np.ndarray,
    target_names: list[str],
    feature_names: list[str],
    holdout_values: np.ndarray,
    results: pd.DataFrame,
) -> None:
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    lines = [
        "# Сводка запуска lab5",
        "",
        "Датасет: 20 Newsgroups из `sklearn.datasets.fetch_20newsgroups`, subset=train.",
        f"Категории: {', '.join(target_names)}.",
        f"Документы: {matrix.shape[0]}, термины после TF-IDF: {matrix.shape[1]}, ненулевых значений: {matrix.nnz}.",
        f"Плотность матрицы: {density:.4f}. Скрытых TF-IDF значений для теста: {holdout_values.size}.",
        "Баланс категорий: "
        + ", ".join(f"{name}: {int(np.sum(targets == index))}" for index, name in enumerate(target_names))
        + ".",
        "Примеры терминов: " + ", ".join(feature_names[:15]) + ".",
        "",
        "## Результаты",
        "",
        f"| Модель | RMSE | NDCG@{NDCG_K} | Fit time, sec | Iterations/components |",
        "|--------|------|---------|---------------|-----------------------|",
    ]

    for _, row in results.iterrows():
        lines.append(
            "| {model} | {rmse:.4f} | {ndcg:.4f} | {time:.3f} | {n_iter} |".format(
                model=row["model"],
                rmse=row["rmse"],
                ndcg=row[f"ndcg@{NDCG_K}"],
                time=row["fit_time_sec"],
                n_iter=int(row["n_iter"]),
            )
        )

    (ARTIFACTS_DIR / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    matrix, targets, target_names, feature_names = load_dataset()
    train_matrix, holdout_rows, holdout_columns, holdout_values = make_holdout(matrix)
    results, fitted = run_experiment(train_matrix, holdout_rows, holdout_columns, holdout_values)
    save_results(results, fitted)
    save_run_summary(matrix, targets, target_names, feature_names, holdout_values, results)
    print((ARTIFACTS_DIR / "run_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
