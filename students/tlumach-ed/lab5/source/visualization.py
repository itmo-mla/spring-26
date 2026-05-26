import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_rating_distribution(df, save=True):
    _ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df["rating"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color="#2196F3", edgecolor="white",
                  linewidth=0.5, alpha=0.85)

    ax.set_xlabel("Рейтинг", fontsize=13)
    ax.set_ylabel("Количество оценок", fontsize=13)
    ax.set_title("Распределение рейтингов — датасет Libimseti", fontsize=14)
    ax.set_xticks(range(1, 11))

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * counts.max(),
                f"{int(h):,}", ha="center", va="bottom", fontsize=9)

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "rating_distribution.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_user_activity(df, save=True):
    _ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(8, 5))

    user_counts = df.groupby("user_idx").size()
    ax.hist(user_counts, bins=50, color="#4CAF50", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Число оценок пользователя", fontsize=13)
    ax.set_ylabel("Количество пользователей (log)", fontsize=13)
    ax.set_title("Активность пользователей — датасет Libimseti", fontsize=14)
    ax.axvline(user_counts.median(), color="red", linestyle="--",
               label=f"Медиана: {user_counts.median():.0f}")
    ax.legend(fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "user_activity.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rmse_comparison(results, save=True):
    _ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(9, 5))

    models = list(results.keys())
    values = list(results.values())
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"][:len(models)]

    bars = ax.bar(models, values, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("RMSE (ниже = лучше)", fontsize=13)
    ax.set_title("Сравнение моделей по RMSE", fontsize=14)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "rmse_comparison.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ndcg_comparison(results, k=10, save=True):
    _ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(9, 5))

    models = list(results.keys())
    values = list(results.values())
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"][:len(models)]

    bars = ax.bar(models, values, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel(f"NDCG@{k} (выше = лучше)", fontsize=13)
    ax.set_title(f"Сравнение моделей по NDCG@{k}", fontsize=14)
    ax.set_ylim(0, min(1.0, max(values) * 1.2) if values else 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"ndcg_{k}_comparison.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lfm_training_curve(lfm_model, save=True):
    _ensure_plots_dir()
    history = lfm_model.train_rmse_history
    if not history:
        print("Нет истории обучения LFM.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history) + 1)
    ax.plot(epochs, history, color="#2196F3", linewidth=2,
            marker="o", markersize=4, label="Train RMSE (LFM собственный)")
    ax.set_xlabel("Эпоха", fontsize=13)
    ax.set_ylabel("RMSE", fontsize=13)
    ax.set_title("Кривая обучения LFM (SGD)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "lfm_training_curve.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_slim_weight_sparsity(slim_model, save=True):
    _ensure_plots_dir()
    W = slim_model.W
    if W is None:
        print("SLIM не обучен.")
        return

    nonzero_per_item = np.diff(W.indptr)  # число ненулевых весов в каждой строке

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nonzero_per_item, bins=40, color="#FF5722", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.set_xlabel("Число ненулевых весов на айтем", fontsize=13)
    ax.set_ylabel("Количество айтемов", fontsize=13)
    ax.set_title("Разреженность матрицы весов SLIM", fontsize=14)
    ax.axvline(np.median(nonzero_per_item), color="navy", linestyle="--",
               label=f"Медиана: {np.median(nonzero_per_item):.1f}")
    ax.legend(fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "slim_weight_sparsity.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)