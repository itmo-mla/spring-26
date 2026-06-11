from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_splits
from tree import ID3GiniClassifier


LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"


def evaluate_model(name: str, y_true, y_pred) -> dict[str, float | str]:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def make_sklearn_tree() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "tree",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=7,
                    min_samples_split=24,
                    min_samples_leaf=8,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data = make_splits()

    tree = ID3GiniClassifier(max_depth=7, min_samples_split=24, min_samples_leaf=8)
    tree.fit(data.X_train, data.y_train, data.feature_types)
    y_pred_before = tree.predict(data.X_test)
    before_stats = tree.stats()
    (ARTIFACTS_DIR / "tree_before_pruning.txt").write_text(tree.export_text(max_depth=5), encoding="utf-8")

    pruned_nodes = tree.prune(data.X_val, data.y_val)
    y_pred_after = tree.predict(data.X_test)
    after_stats = tree.stats()
    (ARTIFACTS_DIR / "tree_after_pruning.txt").write_text(tree.export_text(max_depth=5), encoding="utf-8")

    sklearn_tree = make_sklearn_tree()
    sklearn_tree.fit(data.X_train, data.y_train)
    y_pred_sklearn = sklearn_tree.predict(data.X_test)

    metrics = pd.DataFrame(
        [
            evaluate_model("ID3 before pruning", data.y_test, y_pred_before),
            evaluate_model("ID3 after pruning", data.y_test, y_pred_after),
            evaluate_model("sklearn DecisionTreeClassifier", data.y_test, y_pred_sklearn),
        ]
    )
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)

    summary = {
        "dataset": data.source_name,
        "sizes": {
            "train": len(data.X_train),
            "validation": len(data.X_val),
            "test": len(data.X_test),
        },
        "missing_values": data.missing_summary.astype(int).to_dict(),
        "tree_before_pruning": before_stats,
        "tree_after_pruning": after_stats,
        "pruned_nodes": pruned_nodes,
        "confusion_matrices": {
            "id3_before_pruning": confusion_matrix(data.y_test, y_pred_before).tolist(),
            "id3_after_pruning": confusion_matrix(data.y_test, y_pred_after).tolist(),
            "sklearn": confusion_matrix(data.y_test, y_pred_sklearn).tolist(),
        },
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Dataset: {data.source_name}")
    print("Missing values:")
    print(data.missing_summary.to_string())
    print("\nMetrics:")
    print(metrics.round(4).to_string(index=False))
    print(f"\nPruned nodes: {pruned_nodes}")
    print(f"Before pruning: {before_stats}")
    print(f"After pruning: {after_stats}")


if __name__ == "__main__":
    main()
