from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_splits
from metrics import (
    evaluate_classifier,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_metric_bars,
    plot_tree_complexity,
)
from tree import ProbabilisticID3Classifier


LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"


def build_sklearn_tree(random_state: int, max_depth: int, min_samples_leaf: int) -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_report(
    metrics: pd.DataFrame,
    before_stats: dict[str, int],
    after_stats: dict[str, int],
    pruned_nodes: int,
    sample_size: int,
) -> str:
    table = metrics.copy()
    for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")

    metric_lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.to_dict("records"):
        metric_lines.append(
            "| {model} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} |".format(**row)
        )

    lines = [
        "# Experiment summary",
        "",
        f"Sample size: {sample_size}",
        f"Pruned internal nodes: {pruned_nodes}",
        "",
        "## Metrics",
        "",
        *metric_lines,
        "",
        "## Tree complexity",
        "",
        "| State | Depth | Nodes | Leaves |",
        "|---|---:|---:|---:|",
        f"| Before pruning | {before_stats['depth']} | {before_stats['nodes']} | {before_stats['leaves']} |",
        f"| After pruning | {after_stats['depth']} | {after_stats['nodes']} | {after_stats['leaves']} |",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 1: custom ID3 decision tree on Adult Income")
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--min-samples-split", type=int, default=40)
    parser.add_argument("--min-samples-leaf", type=int, default=15)
    parser.add_argument("--min-gain", type=float, default=1e-4)
    parser.add_argument("--max-thresholds", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = make_splits(sample_size=args.sample_size, random_state=args.random_state)

    custom_tree = ProbabilisticID3Classifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        min_gain=args.min_gain,
        max_thresholds=args.max_thresholds,
    )
    custom_tree.fit(bundle.X_train, bundle.y_train, bundle.feature_types)
    before_stats = custom_tree.stats()
    before_metrics = evaluate_classifier("Custom ID3 before pruning", custom_tree, bundle.X_test, bundle.y_test)

    before_preview = custom_tree.export_text(max_depth=4)
    custom_tree_before = copy.deepcopy(custom_tree)
    pruned_nodes = custom_tree.prune(bundle.X_val, bundle.y_val)
    after_stats = custom_tree.stats()
    after_metrics = evaluate_classifier("Custom ID3 after pruning", custom_tree, bundle.X_test, bundle.y_test)
    after_preview = custom_tree.export_text(max_depth=4)

    sklearn_tree = build_sklearn_tree(
        random_state=args.random_state,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    sklearn_tree.fit(bundle.X_train, bundle.y_train)
    sklearn_metrics = evaluate_classifier("Sklearn DecisionTree", sklearn_tree, bundle.X_test, bundle.y_test)

    metrics = pd.DataFrame([before_metrics, after_metrics, sklearn_metrics])
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)
    custom_tree.feature_importances_.to_csv(ARTIFACTS_DIR / "feature_importances.csv", header=["importance"])

    (ARTIFACTS_DIR / "tree_before_pruning.txt").write_text(before_preview + "\n", encoding="utf-8")
    (ARTIFACTS_DIR / "tree_after_pruning.txt").write_text(after_preview + "\n", encoding="utf-8")

    summary = {
        "dataset": bundle.source_name,
        "sample_size": args.sample_size,
        "train_size": len(bundle.X_train),
        "validation_size": len(bundle.X_val),
        "test_size": len(bundle.X_test),
        "missing_values": bundle.missing_summary.to_dict(),
        "class_balance": bundle.class_balance.to_dict(),
        "before_pruning": before_stats,
        "after_pruning": after_stats,
        "pruned_internal_nodes": pruned_nodes,
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / "experiment_summary.md").write_text(
        make_report(metrics, before_stats, after_stats, pruned_nodes, args.sample_size),
        encoding="utf-8",
    )

    plot_metric_bars(metrics, ARTIFACTS_DIR / "metrics.png")
    plot_confusion_matrices(
        {
            "Custom before": custom_tree_before,
            "Custom after": custom_tree,
            "Sklearn": sklearn_tree,
        },
        bundle.X_test,
        bundle.y_test,
        ARTIFACTS_DIR / "confusion_matrices.png",
    )
    plot_tree_complexity(before_stats, after_stats, ARTIFACTS_DIR / "tree_complexity.png")
    plot_feature_importance(custom_tree.feature_importances_, ARTIFACTS_DIR / "feature_importance.png")

    print(make_report(metrics, before_stats, after_stats, pruned_nodes, args.sample_size))
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
