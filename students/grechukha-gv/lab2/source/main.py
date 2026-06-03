import argparse
import csv
import json
import time
from typing import Iterable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    )
from sklearn.model_selection import ParameterGrid

from data import artifacts_dir, load_chronic_kidney_disease_uci
from plots import (
    plot_confusion_matrices,
    plot_learning_curve,
    plot_oob_permutation_importance,
    plot_roc_curves,
    )
from random_forest import RandomForestClassifier


def grid_search_oob(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict,
    random_state: int,
    grid_csv_path,
    ) -> tuple[dict, float]:
    """Перебор по сетке. Лучшая комбинация - по OOB accuracy кастомного леса"""
    best_params: dict = {}
    best_oob = -1.0
    rows: list[dict] = []
    for params in ParameterGrid(param_grid):
        model = RandomForestClassifier(random_state=random_state, **params)
        model.fit(X, y)
        rows.append({**params, "oob_score": float(model.oob_score_)})
        if model.oob_score_ > best_oob:
            best_oob = float(model.oob_score_)
            best_params = dict(params)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(grid_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return best_params, best_oob


def time_fit(
    model_factory, X: np.ndarray, y: np.ndarray, n_runs: int
    ) -> tuple[float, float]:
    """Возвращает (mean, std) времени обучения по n_runs независимым прогонам.
    Фабрика возвращает новую необученную модель на каждый вызов"""
    times = []
    for _ in range(n_runs):
        model = model_factory()
        t0 = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    return float(arr.mean()), float(arr.std())


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    positive_label: int,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_score)) if y_score is not None else float("nan")
    )
    return metrics


def fmt_metrics(name: str, m: dict, time_mean: float, time_std: float) -> str:
    return (
        f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} "
        f"| {m['f1']:.4f} | {m['roc_auc']:.4f} | {time_mean:.4f} ± {time_std:.4f} |"
    )


def learning_curve_by_n_estimators(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_values: Iterable[int],
    base_params: dict,
    random_state: int,
    ) -> tuple[list[int], list[float], list[float]]:
    """Зависимость OOB accuracy (train) и test accuracy от числа деревьев.
    Параметры базовой модели - без n_estimators"""
    n_list = list(n_values)
    oob_scores: list[float] = []
    test_scores: list[float] = []
    for n in n_list:
        model = RandomForestClassifier(
            n_estimators=n, random_state=random_state, **base_params
        )
        model.fit(X_train, y_train)
        oob_scores.append(float(model.oob_score_))
        test_scores.append(float(np.mean(model.predict(X_test) == y_test)))
    return n_list, oob_scores, test_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab2: Random Forest + OOB grid search")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed для воспроизводимости"
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Не выполнять полный grid search (использовать параметры по умолчанию)",
    )
    parser.add_argument(
        "--n-time-runs",
        type=int,
        default=5,
        help="Сколько повторов делать для замера времени обучения",
    )
    parser.add_argument(
        "--n-permutation-repeats",
        type=int,
        default=20,
        help="Сколько перестановок усреднять в OOB^j",
    )
    parser.add_argument(
        "--skip-learning-curve",
        action="store_true",
        help="Пропустить построение кривой по n_estimators",
    )
    args = parser.parse_args()
    rs = args.random_state

    X_train, X_test, y_train, y_test, feature_names = load_chronic_kidney_disease_uci(
        random_state=rs
    )
    art = artifacts_dir()

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_features": ["sqrt", "log2", 0.5],
        "max_depth": [None, 8, 12],
        "min_samples_split": [2, 5],
    }

    if args.skip_grid:
        best_params = {
            "n_estimators": 100,
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_split": 2,
        }
        probe = RandomForestClassifier(random_state=rs, **best_params)
        probe.fit(X_train, y_train)
        best_oob = float(probe.oob_score_)
    else:
        best_params, best_oob = grid_search_oob(
            X_train,
            y_train,
            param_grid,
            random_state=rs,
            grid_csv_path=art / "grid_results.csv",
        )

    (art / "best_params.json").write_text(
        json.dumps({"best_params": best_params, "best_oob_grid": best_oob}, indent=2),
        encoding="utf-8",
    )

    custom_factory = lambda: RandomForestClassifier(random_state=rs, **best_params)
    sk_factory = lambda: SklearnRF(
        **best_params,
        random_state=rs,
        n_jobs=1,
        bootstrap=True,
        oob_score=True,
    )

    time_custom_mean, time_custom_std = time_fit(
        custom_factory, X_train, y_train, args.n_time_runs
    )
    time_sklearn_mean, time_sklearn_std = time_fit(
        sk_factory, X_train, y_train, args.n_time_runs
    )

    custom = custom_factory()
    custom.fit(X_train, y_train)
    sk = sk_factory()
    sk.fit(X_train, y_train)

    classes = np.sort(np.unique(y_train))
    positive_label = int(classes.max())

    proba_c = custom.predict_proba(X_test)
    proba_s = sk.predict_proba(X_test)
    pos_col_c = int(np.where(custom.classes_ == positive_label)[0][0])
    pos_col_s = int(np.where(sk.classes_ == positive_label)[0][0])
    y_pred_c = custom.classes_[np.argmax(proba_c, axis=1)]
    y_pred_s = sk.classes_[np.argmax(proba_s, axis=1)]

    metrics_c = binary_metrics(y_test, y_pred_c, proba_c[:, pos_col_c], positive_label)
    metrics_s = binary_metrics(y_test, y_pred_s, proba_s[:, pos_col_s], positive_label)

    dummy = DummyClassifier(strategy="most_frequent", random_state=rs)
    t0 = time.perf_counter()
    dummy.fit(X_train, y_train)
    time_dummy = time.perf_counter() - t0
    proba_d = dummy.predict_proba(X_test)
    pos_col_d = int(np.where(dummy.classes_ == positive_label)[0][0])
    metrics_d = binary_metrics(
        y_test, dummy.predict(X_test), proba_d[:, pos_col_d], positive_label
    )

    custom.compute_oob_permutation_importance(
        X_train,
        y_train,
        n_repeats=args.n_permutation_repeats,
        random_state=rs,
    )
    imp = custom.oob_permutation_importances_
    imp_std = custom.oob_permutation_importances_std_
    plot_oob_permutation_importance(
        feature_names,
        imp,
        imp_std,
        art / "oob_permutation_importance.png",
        title=(
            f"Важность признаков (OOB^j, среднее по {args.n_permutation_repeats} "
            "перестановкам ± std)"
        ),
    )

    plot_confusion_matrices(
        {
            "Custom RF": confusion_matrix(y_test, y_pred_c, labels=classes),
            "Sklearn RF": confusion_matrix(y_test, y_pred_s, labels=classes),
        },
        class_labels=[str(c) for c in classes],
        out_path=art / "confusion_matrices.png",
    )

    fpr_c, tpr_c, _ = roc_curve(
        y_test, proba_c[:, pos_col_c], pos_label=positive_label
    )
    fpr_s, tpr_s, _ = roc_curve(
        y_test, proba_s[:, pos_col_s], pos_label=positive_label
    )
    plot_roc_curves(
        {
            "Custom RF": (fpr_c, tpr_c, metrics_c["roc_auc"]),
            "Sklearn RF": (fpr_s, tpr_s, metrics_s["roc_auc"]),
        },
        out_path=art / "roc_curves.png",
    )

    if not args.skip_learning_curve:
        base_for_curve = {k: v for k, v in best_params.items() if k != "n_estimators"}
        n_values = [10, 25, 50, 100, 200, 400]
        n_list, oob_scores, test_scores = learning_curve_by_n_estimators(
            X_train, y_train, X_test, y_test, n_values, base_for_curve, rs
        )
        plot_learning_curve(
            n_list,
            oob_scores,
            test_scores,
            out_path=art / "learning_curve.png",
        )
    else:
        n_list, oob_scores, test_scores = [], [], []

    lines = [
        "# Результаты запуска (генерируется main.py)",
        "",
        f"- Лучшие параметры (по OOB на сетке train): `{best_params}`",
        f"- Лучший OOB на сетке (custom, train): {best_oob:.4f}",
        f"- OOB после финального fit (custom, train): {custom.oob_score_:.4f}",
        f"- OOB sklearn (train): {float(sk.oob_score_):.4f}",
        "",
        f"Время обучения усреднено по {args.n_time_runs} прогонам, OOB^j — "
        f"по {args.n_permutation_repeats} перестановкам.",
        "",
        "## Качество на тесте",
        "",
        "| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Время обучения, с |",
        "|---|---:|---:|---:|---:|---:|---:|",
        fmt_metrics("Custom RF", metrics_c, time_custom_mean, time_custom_std),
        fmt_metrics("Sklearn RF", metrics_s, time_sklearn_mean, time_sklearn_std),
        fmt_metrics("DummyClassifier (most_frequent)", metrics_d, time_dummy, 0.0),
        "",
        f"Positive class = {positive_label} (минорный/мажорный определяется датасетом).",
        "",
        "## Топ-10 OOB^j (mean ± std)",
        "",
    ]
    top_idx = np.argsort(imp)[::-1][:10]
    for rank, j in enumerate(top_idx, start=1):
        lines.append(
            f"{rank}. `{feature_names[j]}` — {imp[j]:.4f} ± {imp_std[j]:.4f}"
        )

    if n_list:
        lines += [
            "",
            "## Зависимость качества от n_estimators",
            "",
            "| n_estimators | OOB (train) | Accuracy (test) |",
            "|---:|---:|---:|",
        ]
        for n, o, t in zip(n_list, oob_scores, test_scores):
            lines.append(f"| {n} | {o:.4f} | {t:.4f} |")

    (art / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"best_params": best_params, "best_oob_grid": best_oob}, indent=2))
    print(
        f"custom: acc={metrics_c['accuracy']:.4f} f1={metrics_c['f1']:.4f} "
        f"auc={metrics_c['roc_auc']:.4f} time={time_custom_mean:.4f}±{time_custom_std:.4f}s"
    )
    print(
        f"sklearn: acc={metrics_s['accuracy']:.4f} f1={metrics_s['f1']:.4f} "
        f"auc={metrics_s['roc_auc']:.4f} time={time_sklearn_mean:.4f}±{time_sklearn_std:.4f}s"
    )
    print(
        f"dummy:  acc={metrics_d['accuracy']:.4f} f1={metrics_d['f1']:.4f} "
        f"auc={metrics_d['roc_auc']:.4f} time={time_dummy:.4f}s"
    )


if __name__ == "__main__":
    main()
