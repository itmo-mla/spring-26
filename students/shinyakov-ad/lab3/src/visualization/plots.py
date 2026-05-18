from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cv_metrics(summary: pd.DataFrame, output_path) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available_metrics = [col for col in metric_cols if col in summary.columns]
    labels = summary["model"].to_list()
    x = np.arange(len(labels))
    width = 0.14

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, metric in enumerate(available_metrics):
        offset = (idx - (len(available_metrics) - 1) / 2) * width
        ax.bar(x + offset, summary[metric].astype(float), width=width, label=metric)

    ax.set_title("Cross-validation metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fit_time(summary: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["model"], summary["fit_time_sec"], color=["#4477aa", "#cc6677"][: len(summary)])
    ax.set_title("Average fit time per CV fold")
    ax.set_ylabel("seconds")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
