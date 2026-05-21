import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA


def load_data():
    data = load_wine()
    return data.data.astype(float), list(data.feature_names)


def evaluate_gmm(model, X_train: np.ndarray, X_test: np.ndarray, label: str, save_path: str | None = None) -> dict:
    train_ll = model.score(X_train)
    test_ll = model.score(X_test)
    bic_val = model.bic(X_test) if hasattr(model, "bic") else float("nan")
    aic_val = model.aic(X_test) if hasattr(model, "aic") else float("nan")

    report = (
        f"{label}\n"
        f"Log-likelihood (train): {train_ll:.4f}, Log-likelihood (test): {test_ll:.4f}, "
        f"BIC: {bic_val:.2f}, AIC: {aic_val:.2f}"
    )
    logging.info("\n" + report)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")

    return {"train_ll": train_ll, "test_ll": test_ll, "bic": bic_val, "aic": aic_val}


def plot_pca_clusters(X: np.ndarray, labels_custom: np.ndarray, labels_sklearn: np.ndarray, save_path: str):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    K = len(np.unique(labels_custom))
    cmap = plt.cm.get_cmap("tab10", K)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, labels, title in zip(axes, [labels_custom, labels_sklearn], ["Custom GMM", "sklearn GMM"]):
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap, vmin=-0.5, vmax=K - 0.5, s=20, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(sc, ax=ax, ticks=range(K), label="Компонента")

    plt.suptitle("Кластеризация GMM (PCA 2D)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


