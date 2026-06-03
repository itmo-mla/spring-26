from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from boosting import LogisticGradientBoostingClassifier
from data import load_cancer_dataset
from metrics import (
    evaluate_classifier,
    plot_confusion_matrices,
    plot_cv_metrics,
    plot_feature_importances,
    plot_learning_curve,
    plot_training_time,
    summarize_cv,
)


LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"


def parse_max_features(value: str | None) -> int | float | str | None:
    if value is None or value.lower() == "none":
        return None
    if value in {"sqrt", "log2"}:
        return value
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max_features must be None, sqrt, log2, an integer, or a float"
        ) from exc


def make_custom_model(args: argparse.Namespace, random_state: int) -> LogisticGradientBoostingClassifier:
    return LogisticGradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        max_features=args.max_features,
        random_state=random_state,
    )


def make_sklearn_model(args: argparse.Namespace, random_state: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        max_features=args.max_features,
        random_state=random_state,
    )


def run_cross_validation(bundle, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    rows: list[dict[str, float | str]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(bundle.X_train, bundle.y_train), start=1):
        X_fold_train = bundle.X_train.iloc[train_idx].reset_index(drop=True)
        X_fold_val = bundle.X_train.iloc[val_idx].reset_index(drop=True)
        y_fold_train = bundle.y_train[train_idx]
        y_fold_val = bundle.y_train[val_idx]

        for name, factory in [
            ("Custom Gradient Boosting", make_custom_model),
            ("Sklearn GradientBoosting", make_sklearn_model),
        ]:
            model = factory(args, args.random_state + fold)
            started = time.perf_counter()
            model.fit(X_fold_train, y_fold_train)
            elapsed = time.perf_counter() - started
            row = evaluate_classifier(name, model, X_fold_val, y_fold_val, elapsed)
            row["fold"] = fold
            rows.append(row)

    cv_results = pd.DataFrame(rows)
    return cv_results, summarize_cv(rows)


def fit_final_models(bundle, args: argparse.Namespace):
    custom_model = make_custom_model(args, args.random_state)
    started = time.perf_counter()
    custom_model.fit(bundle.X_train, bundle.y_train)
    custom_time = time.perf_counter() - started

    sklearn_model = make_sklearn_model(args, args.random_state)
    started = time.perf_counter()
    sklearn_model.fit(bundle.X_train, bundle.y_train)
    sklearn_time = time.perf_counter() - started

    test_metrics = pd.DataFrame(
        [
            evaluate_classifier("Custom Gradient Boosting", custom_model, bundle.X_test, bundle.y_test, custom_time),
            evaluate_classifier("Sklearn GradientBoosting", sklearn_model, bundle.X_test, bundle.y_test, sklearn_time),
        ]
    )
    return custom_model, sklearn_model, test_metrics


def markdown_cv_table(cv_summary: pd.DataFrame) -> list[str]:
    rows = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Log loss | Train time, s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cv_summary.to_dict("records"):
        rows.append(
            "| {model} | {accuracy_mean:.4f} ± {accuracy_std:.4f} | "
            "{precision_mean:.4f} ± {precision_std:.4f} | "
            "{recall_mean:.4f} ± {recall_std:.4f} | "
            "{f1_mean:.4f} ± {f1_std:.4f} | "
            "{roc_auc_mean:.4f} ± {roc_auc_std:.4f} | "
            "{log_loss_mean:.4f} ± {log_loss_std:.4f} | "
            "{train_time_sec_mean:.4f} |".format(**row)
        )
    return rows


def markdown_test_table(test_metrics: pd.DataFrame) -> list[str]:
    rows = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Log loss | Train time, s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_metrics.to_dict("records"):
        rows.append(
            "| {model} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
            "{f1:.4f} | {roc_auc:.4f} | {log_loss:.4f} | {train_time_sec:.4f} |".format(**row)
        )
    return rows


def make_report(bundle, args, cv_summary: pd.DataFrame, test_metrics: pd.DataFrame) -> str:
    lines = [
        "# Experiment summary",
        "",
        f"Dataset: {bundle.description['name']}",
        f"Samples: {bundle.description['samples']}",
        f"Features: {bundle.description['features']}",
        f"Positive class: {bundle.description['positive_class']}",
        "",
        "## Model parameters",
        "",
        f"`n_estimators={args.n_estimators}`, `learning_rate={args.learning_rate}`, "
        f"`max_depth={args.max_depth}`, `subsample={args.subsample}`, "
        f"`min_samples_leaf={args.min_samples_leaf}`",
        "",
        "## Cross-validation",
        "",
        *markdown_cv_table(cv_summary),
        "",
        "## Hold-out test",
        "",
        *markdown_test_table(test_metrics),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 3: custom Gradient Boosting")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=140)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-samples-split", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--max-features", type=parse_max_features, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_cancer_dataset(test_size=args.test_size, random_state=args.random_state)
    cv_results, cv_summary = run_cross_validation(bundle, args)
    custom_model, sklearn_model, test_metrics = fit_final_models(bundle, args)

    cv_results.to_csv(ARTIFACTS_DIR / "cv_results.csv", index=False)
    cv_summary.to_csv(ARTIFACTS_DIR / "cv_summary.csv", index=False)
    test_metrics.to_csv(ARTIFACTS_DIR / "test_metrics.csv", index=False)

    importance_frame = pd.DataFrame(
        {
            "feature": bundle.feature_names,
            "custom_importance": custom_model.feature_importances_,
            "sklearn_importance": sklearn_model.feature_importances_,
        }
    ).sort_values("custom_importance", ascending=False)
    importance_frame.to_csv(ARTIFACTS_DIR / "feature_importances.csv", index=False)

    summary = {
        "dataset": bundle.description,
        "train_size": len(bundle.X_train),
        "test_size": len(bundle.X_test),
        "parameters": vars(args),
        "custom_final_train_loss": custom_model.train_loss_[-1],
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = make_report(bundle, args, cv_summary, test_metrics)
    (ARTIFACTS_DIR / "experiment_summary.md").write_text(report, encoding="utf-8")

    plot_cv_metrics(cv_summary, ARTIFACTS_DIR / "cv_metrics.png", args.cv_folds)
    plot_training_time(cv_summary, test_metrics, ARTIFACTS_DIR / "training_time.png")
    plot_learning_curve(custom_model.train_loss_, ARTIFACTS_DIR / "learning_curve.png")
    plot_confusion_matrices(
        {"Custom GB": custom_model, "Sklearn GB": sklearn_model},
        bundle.X_test,
        bundle.y_test,
        ARTIFACTS_DIR / "confusion_matrices.png",
    )
    plot_feature_importances(
        bundle.feature_names,
        custom_model.feature_importances_,
        sklearn_model.feature_importances_,
        ARTIFACTS_DIR / "feature_importances.png",
    )

    print(report)
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
