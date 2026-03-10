from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data_load import load_dataset
from model import DecisionTree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def evaluate_model(model, X, y):
    y_pred = np.asarray(model.predict(X))
    y_proba = np.asarray(model.predict_proba(X))[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y, y_proba)
    except ValueError:
        metrics["roc_auc"] = np.nan

    cm = confusion_matrix(y, y_pred)
    return metrics, cm


def print_metrics(name, metrics):
    print(f"\n{name}")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        value = metrics[key]
        if np.isnan(value):
            print(f"  {key}: n/a")
        else:
            print(f"  {key}: {value:.4f}")


def plot_classification_metrics(results, output_path):
    labels = [row["model"] for row in results]
    x = np.arange(len(labels))
    width = 0.22

    accuracy = np.array([row["accuracy"] for row in results], dtype=float)
    roc_auc = np.array([row["roc_auc"] for row in results], dtype=float)
    precision = np.array([row["precision"] for row in results], dtype=float)
    recall = np.array([row["recall"] for row in results], dtype=float)
    f1 = np.array([row["f1"] for row in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_left, ax_right = axes

    ax_left.bar(x - width / 2, accuracy, width=width, label="accuracy")
    ax_left.bar(x + width / 2, np.nan_to_num(roc_auc, nan=0.0), width=width, label="roc_auc")
    ax_left.set_title("Global Metrics (test)")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=15, ha="right")
    ax_left.set_ylim(0, 1.05)
    ax_left.legend()
    ax_left.grid(axis="y", alpha=0.3)

    ax_right.bar(x - width, precision, width=width, label="precision")
    ax_right.bar(x, recall, width=width, label="recall")
    ax_right.bar(x + width, f1, width=width, label="f1")
    small_max = max(0.1, float(np.max([precision.max(), recall.max(), f1.max()])) + 0.05)
    ax_right.set_ylim(0, min(1.05, small_max))
    ax_right.set_title("Minority-Class Metrics (test)")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(labels, rotation=15, ha="right")
    ax_right.legend()
    ax_right.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(confusion_map, output_path):
    n = len(confusion_map)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for idx, (name, cm) in enumerate(confusion_map.items()):
        ax = axes[idx]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_roc_curves(y_true, proba_map, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, y_proba in proba_map.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curves (test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train_full, X_test, y_train_full, y_test, _ = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    custom_tree = DecisionTree(criterion="gini")
    custom_tree.fit(X_train, y_train)
    custom_before_metrics, custom_before_cm = evaluate_model(custom_tree, X_test, y_test)
    custom_before_proba = np.asarray(custom_tree.predict_proba(X_test))[:, 1]

    custom_tree_pruned = deepcopy(custom_tree)
    custom_tree_pruned.prune_inner(X_val, y_val)
    custom_after_metrics, custom_after_cm = evaluate_model(custom_tree_pruned, X_test, y_test)
    custom_after_proba = np.asarray(custom_tree_pruned.predict_proba(X_test))[:, 1]

    sklearn_tree = DecisionTreeClassifier(criterion="gini", random_state=42)
    sklearn_tree.fit(X_train, y_train)
    sklearn_metrics, sklearn_cm = evaluate_model(sklearn_tree, X_test, y_test)
    sklearn_proba = np.asarray(sklearn_tree.predict_proba(X_test))[:, 1]

    print_metrics("Custom tree (before pruning)", custom_before_metrics)
    print_metrics("Custom tree (after pruning)", custom_after_metrics)
    print_metrics("Sklearn tree", sklearn_metrics)

    results = [
        {"model": "custom_before", **custom_before_metrics},
        {"model": "custom_after", **custom_after_metrics},
        {"model": "sklearn", **sklearn_metrics},
    ]

    print("\nSummary (test):")
    print("model            accuracy  precision  recall   f1       roc_auc")
    for row in results:
        print(
            f"{row['model']:<16} "
            f"{row['accuracy']:.4f}    "
            f"{row['precision']:.4f}     "
            f"{row['recall']:.4f}   "
            f"{row['f1']:.4f}   "
            f"{row['roc_auc']:.4f}"
        )

    plot_classification_metrics(results, artifacts_dir / "classification_metrics_test.png")
    plot_confusion_matrices(
        {
            "Custom before": custom_before_cm,
            "Custom after": custom_after_cm,
            "Sklearn": sklearn_cm,
        },
        artifacts_dir / "confusion_matrices_test.png",
    )
    plot_roc_curves(
        y_test,
        {
            "Custom before": custom_before_proba,
            "Custom after": custom_after_proba,
            "Sklearn": sklearn_proba,
        },
        artifacts_dir / "roc_auc_test.png",
    )

    print(f"Saved plots: {artifacts_dir / 'classification_metrics_test.png'}")
    print(f"Saved plots: {artifacts_dir / 'confusion_matrices_test.png'}")
    print(f"Saved plots: {artifacts_dir / 'roc_auc_test.png'}")


if __name__ == "__main__":
    main()
