from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.boosting import GradientBoostingClassifier
from source.data import build_preprocessor, load_features


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 3,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}


def make_custom_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", GradientBoostingClassifier(**MODEL_PARAMS)),
        ]
    )


def make_sklearn_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                SklearnGradientBoostingClassifier(
                    n_estimators=MODEL_PARAMS["n_estimators"],
                    learning_rate=MODEL_PARAMS["learning_rate"],
                    max_depth=MODEL_PARAMS["max_depth"],
                    min_samples_leaf=MODEL_PARAMS["min_samples_leaf"],
                    subsample=MODEL_PARAMS["subsample"],
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, X_test, y_test, fit_time: float) -> dict:
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "fit_time": float(fit_time),
        "predict_time": float(predict_time),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def fit_timed(model: Pipeline, X_train, y_train):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def run_cross_validation(X, y):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipeline in [
        ("custom", make_custom_pipeline()),
        ("sklearn", make_sklearn_pipeline()),
    ]:
        cv_result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
        )
        results[name] = {
            "fit_time": cv_result["fit_time"].tolist(),
            "score_time": cv_result["score_time"].tolist(),
            **{metric: cv_result[f"test_{metric}"].tolist() for metric in scoring},
        }
    return results


def summarize_cv(values: dict) -> dict:
    return {
        metric: {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
        for metric, scores in values.items()
    }


def fitted_feature_names(model: Pipeline) -> list[str]:
    return list(model.named_steps["preprocessor"].get_feature_names_out())


def plot_class_distribution(dataset_info):
    counts = dataset_info["class_counts"]
    values = [counts.get("0", 0), counts.get("1", 0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Not survived", "Survived"], values, color=["#69788f", "#d08c60"])
    ax.set_title("Titanic class distribution")
    ax.set_ylabel("Passengers")
    for idx, value in enumerate(values):
        ax.text(idx, value + 5, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)


def plot_metric_comparison(custom_metrics, sklearn_metrics):
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    custom = [custom_metrics[key] for key in keys]
    sklearn = [sklearn_metrics[key] for key in keys]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, custom, width, label="Custom GB", color="#376996")
    ax.bar(x + width / 2, sklearn, width, label="sklearn GB", color="#d08c60")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Test metrics comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_comparison.png", dpi=180)
    plt.close(fig)


def plot_cv_comparison(cv_summary):
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    custom = [cv_summary["custom"][key]["mean"] for key in keys]
    custom_err = [cv_summary["custom"][key]["std"] for key in keys]
    sklearn = [cv_summary["sklearn"][key]["mean"] for key in keys]
    sklearn_err = [cv_summary["sklearn"][key]["std"] for key in keys]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, custom, width, yerr=custom_err, capsize=3, label="Custom GB", color="#376996")
    ax.bar(x + width / 2, sklearn, width, yerr=sklearn_err, capsize=3, label="sklearn GB", color="#d08c60")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("5-fold cross-validation metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cv_metric_comparison.png", dpi=180)
    plt.close(fig)


def plot_time_comparison(custom_metrics, sklearn_metrics, cv_summary):
    names = ["Fit test", "Predict test", "Fit CV/fold"]
    custom = [
        custom_metrics["fit_time"],
        custom_metrics["predict_time"],
        cv_summary["custom"]["fit_time"]["mean"],
    ]
    sklearn = [
        sklearn_metrics["fit_time"],
        sklearn_metrics["predict_time"],
        cv_summary["sklearn"]["fit_time"]["mean"],
    ]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, custom, width, label="Custom GB", color="#376996")
    ax.bar(x + width / 2, sklearn, width, label="sklearn GB", color="#d08c60")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Seconds")
    ax.set_title("Training and prediction time")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "time_comparison.png", dpi=180)
    plt.close(fig)


def plot_confusions(y_test, custom_metrics, sklearn_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, title, metrics in [
        (axes[0], "Custom GB", custom_metrics),
        (axes[1], "sklearn GB", sklearn_metrics),
    ]:
        cm = confusion_matrix(y_test, metrics["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def plot_roc(y_test, custom_metrics, sklearn_metrics):
    fig, ax = plt.subplots(figsize=(6, 5))
    for title, metrics, color in [
        ("Custom GB", custom_metrics, "#376996"),
        ("sklearn GB", sklearn_metrics, "#d08c60"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
        ax.plot(fpr, tpr, label=f"{title} AUC={metrics['roc_auc']:.3f}", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=180)
    plt.close(fig)


def plot_learning_curve(custom_model: Pipeline):
    gb = custom_model.named_steps["model"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(gb.train_loss_) + 1), gb.train_loss_, color="#376996", linewidth=2)
    ax.set_title("Custom GB training log-loss")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Log-loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "learning_curve_loss.png", dpi=180)
    plt.close(fig)


def plot_feature_importance(custom_model: Pipeline, sklearn_model: Pipeline):
    feature_names = fitted_feature_names(custom_model)
    custom_importance = custom_model.named_steps["model"].feature_importances_
    sklearn_importance = sklearn_model.named_steps["model"].feature_importances_
    top_idx = np.argsort(sklearn_importance)[::-1][:15]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, title, values, color in [
        (axes[0], "Custom GB", custom_importance, "#376996"),
        (axes[1], "sklearn GB", sklearn_importance, "#d08c60"),
    ]:
        ordered = top_idx[::-1]
        ax.barh([feature_names[i] for i in ordered], values[ordered], color=color)
        ax.set_title(title)
        ax.set_xlabel("Impurity importance")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


def write_report(dataset_info, cv_summary, custom_metrics, sklearn_metrics, custom_model, sklearn_model):
    cv_table = f"""| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time, s/fold |
|---|---:|---:|---:|---:|---:|---:|
| Custom Gradient Boosting | {cv_summary['custom']['accuracy']['mean']:.4f} ± {cv_summary['custom']['accuracy']['std']:.4f} | {cv_summary['custom']['precision']['mean']:.4f} ± {cv_summary['custom']['precision']['std']:.4f} | {cv_summary['custom']['recall']['mean']:.4f} ± {cv_summary['custom']['recall']['std']:.4f} | {cv_summary['custom']['f1']['mean']:.4f} ± {cv_summary['custom']['f1']['std']:.4f} | {cv_summary['custom']['roc_auc']['mean']:.4f} ± {cv_summary['custom']['roc_auc']['std']:.4f} | {cv_summary['custom']['fit_time']['mean']:.4f} |
| sklearn GradientBoostingClassifier | {cv_summary['sklearn']['accuracy']['mean']:.4f} ± {cv_summary['sklearn']['accuracy']['std']:.4f} | {cv_summary['sklearn']['precision']['mean']:.4f} ± {cv_summary['sklearn']['precision']['std']:.4f} | {cv_summary['sklearn']['recall']['mean']:.4f} ± {cv_summary['sklearn']['recall']['std']:.4f} | {cv_summary['sklearn']['f1']['mean']:.4f} ± {cv_summary['sklearn']['f1']['std']:.4f} | {cv_summary['sklearn']['roc_auc']['mean']:.4f} ± {cv_summary['sklearn']['roc_auc']['std']:.4f} | {cv_summary['sklearn']['fit_time']['mean']:.4f} |"""

    test_table = f"""| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time, s | Predict time, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom Gradient Boosting | {custom_metrics['accuracy']:.4f} | {custom_metrics['precision']:.4f} | {custom_metrics['recall']:.4f} | {custom_metrics['f1']:.4f} | {custom_metrics['roc_auc']:.4f} | {custom_metrics['fit_time']:.4f} | {custom_metrics['predict_time']:.4f} |
| sklearn GradientBoostingClassifier | {sklearn_metrics['accuracy']:.4f} | {sklearn_metrics['precision']:.4f} | {sklearn_metrics['recall']:.4f} | {sklearn_metrics['f1']:.4f} | {sklearn_metrics['roc_auc']:.4f} | {sklearn_metrics['fit_time']:.4f} | {sklearn_metrics['predict_time']:.4f} |"""

    feature_names = fitted_feature_names(custom_model)
    custom_importance = custom_model.named_steps["model"].feature_importances_
    sklearn_importance = sklearn_model.named_steps["model"].feature_importances_
    top_rows = []
    for idx in np.argsort(sklearn_importance)[::-1][:12]:
        top_rows.append(f"| {feature_names[idx]} | {custom_importance[idx]:.4f} | {sklearn_importance[idx]:.4f} |")
    importance_table = "\n".join(top_rows)

    report = f"""# Лабораторная работа №3. Градиентный бустинг

## Цель работы
Реализовать собственный алгоритм градиентного бустинга, обучить его на выбранном датасете, оценить качество с помощью кросс-валидации и сравнить с эталонной реализацией `scikit-learn` по качеству и времени обучения.

## Выбранный датасет
Использован датасет Titanic с Kaggle. Это классический датасет для бинарной классификации, в котором по информации о пассажире нужно предсказать, выжил ли он после катастрофы.

Целевая переменная `Survived` задает бинарную классификацию: пассажир выжил (`1`) или не выжил (`0`). Размер данных: {dataset_info['n_samples']} объектов. Распределение классов: не выжил = {dataset_info['class_counts'].get('0', 0)}, выжил = {dataset_info['class_counts'].get('1', 0)}.

Помимо исходных полей были сконструированы дополнительные признаки, которые помогают лучше описать пассажира и его билет:

- `Title` из имени пассажира;
- `Deck` из номера каюты;
- `FamilySize = SibSp + Parch + 1`;
- `IsAlone` как индикатор одиночной поездки;
- `FarePerPerson` как тариф, деленный на размер семьи.

Предобработка включает заполнение пропусков медианой для числовых признаков и самым частым значением для категориальных, затем one-hot кодирование категориальных признаков. В кросс-валидации предобработка выполняется внутри `Pipeline`, поэтому имьютеры и кодировщик обучаются только на train-части каждого фолда.

## Описание алгоритма градиентного бустинга
Градиентный бустинг строит ансамбль последовательно. Каждое новое дерево исправляет ошибки текущей модели, обучаясь на антиградиенте функции потерь. Для бинарной классификации использована логистическая функция потерь.

Начальное приближение задается как логит доли положительного класса:

```text
F0 = log(p / (1 - p))
```

На каждой итерации вычисляются вероятности:

```text
p_i = sigmoid(F(x_i))
```

Затем строятся псевдо-остатки:

```text
r_i = y_i - p_i
```

На этих остатках обучается `DecisionTreeRegressor`, после чего ансамбль обновляется:

```text
F_m(x) = F_(m-1)(x) + learning_rate * h_m(x)
```

Итоговая вероятность класса `1` равна `sigmoid(F(x))`. В собственной реализации вручную написана логика бустинга: инициализация, расчет псевдо-остатков, последовательное обучение деревьев, накопление предсказаний и расчет вероятностей. В качестве базового слабого алгоритма используется `DecisionTreeRegressor` из `sklearn`.

Параметры эксперимента:

```json
{json.dumps(MODEL_PARAMS, ensure_ascii=False, indent=2)}
```

## Кросс-валидация
Для устойчивой оценки качества использована стратифицированная 5-fold кросс-валидация. Сравнение выполнено на одинаковых признаках, одинаковой схеме предобработки и одинаковых основных гиперпараметрах.

{cv_table}

## Оценка на тестовой выборке
Дополнительно данные были разделены на train/test в пропорции 80/20 со стратификацией. Эта оценка нужна для наглядных графиков: ROC-кривых, матриц ошибок и сравнения времени.

{test_table}

## Важность признаков
Важность признаков взята как средняя impurity importance по деревьям ансамбля. Ниже приведены признаки, наиболее важные для эталонной реализации, и соответствующие значения для обеих моделей.

| Feature | Custom GB | sklearn GB |
|---|---:|---:|
{importance_table}

Наиболее важные признаки связаны с полом/обращением пассажира, классом билета, возрастом, стоимостью поездки и семейной структурой. Это совпадает с содержательной интерпретацией датасета Titanic.

## Графики

### Распределение классов
![Class distribution](results/plots/class_distribution.png)

### Сравнение метрик на тестовой выборке
![Metric comparison](results/plots/metric_comparison.png)

### Метрики кросс-валидации
![CV metric comparison](results/plots/cv_metric_comparison.png)

### Сравнение времени
![Time comparison](results/plots/time_comparison.png)

### Матрицы ошибок
![Confusion matrices](results/plots/confusion_matrices.png)

### ROC-кривые
![ROC curves](results/plots/roc_curves.png)

### Кривая обучения собственной модели
![Learning curve](results/plots/learning_curve_loss.png)

### Важность признаков
![Feature importance](results/plots/feature_importance.png)

## Сравнение с эталонной реализацией
Собственная реализация показывает качество, близкое к `GradientBoostingClassifier` из `scikit-learn`. Различия объясняются тем, что в учебной реализации деревья обучаются напрямую на псевдо-остатках `y - p`, тогда как библиотечная реализация содержит дополнительные оптимизации и более точные правила обновления терминальных областей.

По времени обучения реализации сопоставимы, потому что основная вычислительная нагрузка в обоих случаях приходится на построение деревьев. При этом `scikit-learn` обычно быстрее и стабильнее за счет оптимизированного кода библиотеки.

## Выводы
В работе реализован градиентный бустинг для бинарной классификации с логистической функцией потерь. На датасете Titanic собственная реализация достигла качества, сопоставимого с эталонной моделью `scikit-learn`, что подтверждается как тестовой выборкой, так и 5-fold кросс-валидацией.

Эксперимент показал, что градиентный бустинг хорошо подходит для табличных данных с небольшим числом объектов и смешанными типами признаков. Последовательное исправление ошибок позволяет получить устойчивое качество даже на компактном датасете Titanic.
## Запуск
```bash
python source/experiment.py
```
"""
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    (PROJECT_ROOT / "README.md").write_text(report, encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, dataset_info = load_features(PROJECT_ROOT)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv_raw = run_cross_validation(X, y)
    cv_summary = {name: summarize_cv(values) for name, values in cv_raw.items()}

    custom_model, custom_fit_time = fit_timed(make_custom_pipeline(), X_train, y_train)
    sklearn_model, sklearn_fit_time = fit_timed(make_sklearn_pipeline(), X_train, y_train)

    custom_metrics = evaluate(custom_model, X_test, y_test, custom_fit_time)
    sklearn_metrics = evaluate(sklearn_model, X_test, y_test, sklearn_fit_time)

    plot_class_distribution(dataset_info)
    plot_metric_comparison(custom_metrics, sklearn_metrics)
    plot_cv_comparison(cv_summary)
    plot_time_comparison(custom_metrics, sklearn_metrics, cv_summary)
    plot_confusions(y_test, custom_metrics, sklearn_metrics)
    plot_roc(y_test, custom_metrics, sklearn_metrics)
    plot_learning_curve(custom_model)
    plot_feature_importance(custom_model, sklearn_model)

    serializable_custom = {k: v for k, v in custom_metrics.items() if k not in {"y_pred", "y_proba"}}
    serializable_sklearn = {k: v for k, v in sklearn_metrics.items() if k not in {"y_pred", "y_proba"}}
    payload = {
        "dataset": dataset_info,
        "params": MODEL_PARAMS,
        "cv_raw": cv_raw,
        "cv_summary": cv_summary,
        "custom_metrics": serializable_custom,
        "sklearn_metrics": serializable_sklearn,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(payload["cv_summary"]).to_json(RESULTS_DIR / "cv_summary.json", force_ascii=False, indent=2)

    write_report(dataset_info, cv_summary, custom_metrics, sklearn_metrics, custom_model, sklearn_model)

    print("Experiment completed")
    print(f"Custom accuracy: {custom_metrics['accuracy']:.4f}, ROC-AUC: {custom_metrics['roc_auc']:.4f}")
    print(f"sklearn accuracy: {sklearn_metrics['accuracy']:.4f}, ROC-AUC: {sklearn_metrics['roc_auc']:.4f}")
    print(f"Report: {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
