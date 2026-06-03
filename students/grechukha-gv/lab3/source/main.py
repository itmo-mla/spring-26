from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from gradient_boosting import GradientBoostingBinaryClassifier


RANDOM_STATE = 42
N_SPLITS = 5
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def load_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data
    y = dataset.target.to_numpy(dtype=int)
    return x, y


def build_custom_model() -> GradientBoostingBinaryClassifier:
    return GradientBoostingBinaryClassifier(
        n_estimators=100,
        learning_rate=0.15,
        max_depth=3,
        min_samples_split=8,
        min_samples_leaf=5,
        max_thresholds=48,
    )


def build_sklearn_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.15,
        max_depth=3,
        min_samples_split=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }


def cross_validate(x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | int | str]] = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_index].to_numpy()
        x_test = x.iloc[test_index].to_numpy()
        y_train = y[train_index]
        y_test = y[test_index]

        for model_name, factory in (
            ("Собственный GB", build_custom_model),
            ("sklearn GB", build_sklearn_model),
        ):
            model = factory()
            start_time = perf_counter()
            model.fit(x_train, y_train)
            fit_time = perf_counter() - start_time
            metrics = evaluate_model(model, x_test, y_test)
            rows.append(
                {
                    "fold": fold_index,
                    "model": model_name,
                    "fit_time_sec": fit_time,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def train_holdout_models(x: pd.DataFrame, y: np.ndarray):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    fitted_models = {}
    for model_name, factory in (
        ("Собственный GB", build_custom_model),
        ("sklearn GB", build_sklearn_model),
    ):
        model = factory()
        start_time = perf_counter()
        model.fit(x_train.to_numpy(), y_train)
        fit_time = perf_counter() - start_time
        fitted_models[model_name] = {
            "model": model,
            "fit_time_sec": fit_time,
            "metrics": evaluate_model(model, x_test.to_numpy(), y_test),
            "confusion_matrix": confusion_matrix(y_test, model.predict(x_test.to_numpy())),
        }

    return fitted_models, x_train, x_test, y_train, y_test


def save_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(ARTIFACTS_DIR / "cv_results.csv", index=False)
    summary = cv_results.groupby("model").agg(["mean", "std"])
    summary.to_csv(ARTIFACTS_DIR / "cv_summary.csv")
    return summary


def plot_metric_comparison(summary: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    means = summary.loc[:, pd.IndexSlice[metrics, "mean"]]
    means.columns = metrics

    ax = means.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Сравнение качества по 5-fold cross-validation")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0.85, 1.01)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Метрика")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "metrics_comparison.png", dpi=160)
    plt.close()


def plot_fit_time(summary: pd.DataFrame) -> None:
    means = summary.loc[:, ("fit_time_sec", "mean")]
    stds = summary.loc[:, ("fit_time_sec", "std")]

    ax = means.plot(kind="bar", yerr=stds, figsize=(7, 4), rot=0, capsize=4)
    ax.set_title("Среднее время обучения по 5-fold cross-validation")
    ax.set_ylabel("Секунды")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "fit_time.png", dpi=160)
    plt.close()


def plot_confusion_matrices(fitted_models: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, (model_name, result) in zip(axes, fitted_models.items(), strict=True):
        matrix = result["confusion_matrix"]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(model_name)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                ax.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("Матрицы ошибок на holdout-выборке")
    plt.savefig(ARTIFACTS_DIR / "confusion_matrices.png", dpi=160)
    plt.close()


def plot_feature_importance(model: GradientBoostingBinaryClassifier, feature_names: list[str]) -> None:
    importances = model.feature_importances_
    if importances is None:
        return

    top_indices = np.argsort(importances)[-10:]
    top_features = [feature_names[index] for index in top_indices]
    top_values = importances[top_indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features, top_values)
    ax.set_title("Top-10 признаков по собственной модели")
    ax.set_xlabel("Нормированная важность")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "feature_importance.png", dpi=160)
    plt.close()


def format_metric(value: float, std: float | None = None) -> str:
    if std is None:
        return f"{value:.4f}"
    return f"{value:.4f} ± {std:.4f}"


def format_time(value: float, std: float | None = None) -> str:
    if std is None:
        return f"{value:.3f}"
    return f"{value:.3f} ± {std:.3f}"


def build_cv_summary_table(cv_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for model_name in ("Собственный GB", "sklearn GB"):
        row = cv_summary.loc[model_name]
        rows.append(
            {
                "Модель": model_name,
                "Accuracy": format_metric(row[("accuracy", "mean")], row[("accuracy", "std")]),
                "Precision": format_metric(row[("precision", "mean")], row[("precision", "std")]),
                "Recall": format_metric(row[("recall", "mean")], row[("recall", "std")]),
                "F1": format_metric(row[("f1", "mean")], row[("f1", "std")]),
                "Fit time, sec": format_time(row[("fit_time_sec", "mean")], row[("fit_time_sec", "std")]),
            }
        )
    return pd.DataFrame(rows)


def build_holdout_table(fitted_models: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for model_name in ("Собственный GB", "sklearn GB"):
        result = fitted_models[model_name]
        metrics = result["metrics"]
        rows.append(
            {
                "Модель": model_name,
                "Accuracy": format_metric(metrics["accuracy"]),
                "Precision": format_metric(metrics["precision"]),
                "Recall": format_metric(metrics["recall"]),
                "F1": format_metric(metrics["f1"]),
                "Fit time, sec": format_time(result["fit_time_sec"]),
            }
        )
    return pd.DataFrame(rows)


def print_console_summary(
    x: pd.DataFrame,
    y: np.ndarray,
    cv_summary: pd.DataFrame,
    fitted_models: dict[str, dict],
) -> None:
    print("Сводка запуска lab3")
    print(f"Датасет: Breast Cancer Wisconsin, объектов: {x.shape[0]}, признаков: {x.shape[1]}.")
    print(f"Баланс классов: malignant=0: {int(np.sum(y == 0))}, benign=1: {int(np.sum(y == 1))}.")
    print()
    print("Cross-validation, mean ± std")
    print(build_cv_summary_table(cv_summary).to_string(index=False))
    print()
    print("Holdout")
    print(build_holdout_table(fitted_models).to_string(index=False))
    print()


def save_run_summary(
    x: pd.DataFrame,
    y: np.ndarray,
    cv_summary: pd.DataFrame,
    fitted_models: dict[str, dict],
) -> None:
    lines = [
        "# Сводка запуска lab3",
        "",
        f"Датасет: Breast Cancer Wisconsin из `sklearn.datasets`, объектов: {x.shape[0]}, признаков: {x.shape[1]}.",
        f"Баланс классов: malignant=0: {int(np.sum(y == 0))}, benign=1: {int(np.sum(y == 1))}.",
        "",
        "## Cross-validation, mean ± std",
        "",
        "| Модель | Accuracy | Precision | Recall | F1 | Fit time, sec |",
        "|--------|----------|-----------|--------|----|---------------|",
    ]

    for _, row in build_cv_summary_table(cv_summary).iterrows():
        lines.append(
            "| {model} | {accuracy} | {precision} | {recall} | {f1} | {time} |".format(
                model=row["Модель"],
                accuracy=row["Accuracy"],
                precision=row["Precision"],
                recall=row["Recall"],
                f1=row["F1"],
                time=row["Fit time, sec"],
            )
        )

    lines.extend(["", "## Holdout", ""])
    lines.extend(["| Модель | Accuracy | Precision | Recall | F1 | Fit time, sec |", "|--------|----------|-----------|--------|----|---------------|"])

    for _, row in build_holdout_table(fitted_models).iterrows():
        lines.append(
            "| {model} | {accuracy} | {precision} | {recall} | {f1} | {time} |".format(
                model=row["Модель"],
                accuracy=row["Accuracy"],
                precision=row["Precision"],
                recall=row["Recall"],
                f1=row["F1"],
                time=row["Fit time, sec"],
            )
        )

    (ARTIFACTS_DIR / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    x, y = load_dataset()
    cv_results = cross_validate(x, y)
    cv_summary = save_cv_results(cv_results)
    fitted_models, _, _, _, _ = train_holdout_models(x, y)

    plot_metric_comparison(cv_summary)
    plot_fit_time(cv_summary)
    plot_confusion_matrices(fitted_models)
    plot_feature_importance(fitted_models["Собственный GB"]["model"], list(x.columns))
    save_run_summary(x, y, cv_summary, fitted_models)

    print_console_summary(x, y, cv_summary, fitted_models)


if __name__ == "__main__":
    main()
