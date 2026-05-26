import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def _draw_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    cov2 = cov[:2, :2]
    pearson = cov2[0, 1] / (np.sqrt(cov2[0, 0] * cov2[1, 1]) + 1e-10)
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(max(1 - pearson, 0))

    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    scale_x = np.sqrt(cov2[0, 0]) * n_std
    scale_y = np.sqrt(cov2[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean[0], mean[1])
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def plot_clusters(X, labels, means, covariances, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    k = len(means)
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for j in range(k):
        mask = labels == j
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=colors[j], s=15, alpha=0.6,
                   label=f"Компонента {j + 1}")
        # Эллипсоид рассеяния — геометрический смысл Σⱼ
        _draw_ellipse(means[j], covariances[j], ax,
                      n_std=2.0,
                      edgecolor=colors[j],
                      facecolor='none',
                      linewidth=2)
        # Центр компоненты μⱼ
        ax.scatter(means[j, 0], means[j, 1],
                   color=colors[j], marker='X', s=150,
                   edgecolors='black', linewidths=0.8, zorder=5)

    ax.set_xlabel("Признак 1")
    ax.set_ylabel("Признак 2")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранён: {filename}")


def plot_log_likelihood(history, k, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history, color='steelblue', linewidth=2)
    ax.set_xlabel("Итерация EM")
    ax.set_ylabel("Логарифм правдоподобия L(w, θ)")
    ax.set_title(f"Сходимость EM-алгоритма (k={k})")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранён: {filename}")


def plot_log_likelihood_comparison(n_components_list,
                                   scores_custom, scores_sklearn,
                                   filename):
    x = np.arange(len(n_components_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, scores_custom, width,
                   label="Собственная реализация", color="steelblue")
    bars2 = ax.bar(x + width / 2, scores_sklearn, width,
                   label="sklearn GaussianMixture", color="orange")

    # Подписи значений над столбцами
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Число компонент k")
    ax.set_ylabel("Среднее log-правдоподобие (тест)")
    ax.set_title("Сравнение качества: собственная реализация vs sklearn")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in n_components_list])
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранён: {filename}")