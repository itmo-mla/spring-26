from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).parent.parent / "plots"


def _save(name: str) -> Path:
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def learning_curve(train_scores: list[float]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(train_scores) + 1), train_scores, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Train Accuracy")
    ax.set_title("Learning Curve — Custom Gradient Boosting")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save("learning_curve.png")


def cv_comparison(custom_accs: np.ndarray, sklearn_accs: np.ndarray) -> Path:
    labels = ["Custom GB", "Sklearn GB"]
    means = [custom_accs.mean(), sklearn_accs.mean()]
    stds = [custom_accs.std(), sklearn_accs.std()]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=["steelblue", "coral"], width=0.4)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("CV Accuracy (5-fold)")
    ax.set_title("Accuracy Comparison: Custom GB vs Sklearn GB")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    return _save("cv_comparison.png")


def time_comparison(custom_time: float, sklearn_time: float) -> Path:
    labels = ["Custom GB", "Sklearn GB"]
    times = [custom_time, sklearn_time]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, times, color=["steelblue", "coral"], width=0.4)
    ax.set_ylabel("Mean Fold Training Time (s)")
    ax.set_title("Training Time Comparison")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t * 1.02, f"{t:.2f}s", ha="center", fontsize=11)
    fig.tight_layout()
    return _save("time_comparison.png")


def confusion_matrix(cm: np.ndarray, class_names: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Custom Gradient Boosting (full train)")
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    return _save("confusion_matrix.png")
