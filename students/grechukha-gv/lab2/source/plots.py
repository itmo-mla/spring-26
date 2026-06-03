from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_oob_permutation_importance(
    feature_names: list[str],
    importances: np.ndarray,
    stds: np.ndarray,
    out_path: Path,
    top_k: int = 15,
    title: str = "Важность признаков (OOB^j, среднее по 20 перестановкам ± std)",
    ) -> None:
    order = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in order]
    vals = importances[order]
    errs = stds[order]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))
    y_pos = np.arange(len(names))
    ax.barh(
        y_pos,
        vals,
        xerr=errs,
        color="steelblue",
        ecolor="black",
        capsize=3,
        label="Снижение OOB accuracy (mean ± std)",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Снижение OOB accuracy при перестановке признака")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(
    confusion_matrices: dict[str, np.ndarray],
    class_labels: Sequence[str],
    out_path: Path,
    title: str = "Confusion matrices (test)",
    ) -> None:
    n = len(confusion_matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        cm = np.asarray(cm)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        vmax = cm.max() if cm.size else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > vmax / 2 else "black"
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
    title: str = "ROC-кривые (test)",
    ) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random (AUC=0.5)")
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curve(
    n_estimators_list: list[int],
    oob_scores: list[float],
    test_scores: list[float],
    out_path: Path,
    title: str = "Зависимость качества от n_estimators",
    ) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_estimators_list, oob_scores, marker="o", label="OOB accuracy (train)")
    ax.plot(n_estimators_list, test_scores, marker="s", label="Test accuracy")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
