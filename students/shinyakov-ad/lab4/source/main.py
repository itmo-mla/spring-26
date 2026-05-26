from pathlib import Path
import os
import time
import warnings

LAB_DIR = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = LAB_DIR / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import pandas as pd
from data_load import load_dataset
from metrics import aic_score, bic_score, clustering_accuracy
from model import GMM
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


RANDOM_STATE = 42


def count_gmm_params(n_components, n_features):
    return (
        (n_components - 1)
        + n_components * n_features
        + n_components * n_features * (n_features + 1) / 2
    )


def save_clusters_plot(X, labels, output_path, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=25)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    artifacts_dir = LAB_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(random_state=RANDOM_STATE)
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    n_components = len(set(dataset["y_train"]))
    n_features = X_train.shape[1]
    n_params = count_gmm_params(n_components, n_features)

    start_time = time.time()
    custom_gmm = GMM(
        n_components=n_components,
        max_iter=100,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )
    custom_gmm.fit(X_train)
    custom_time = time.time() - start_time

    custom_train_preds = custom_gmm.predict(X_train)
    custom_test_preds = custom_gmm.predict(X_test)
    custom_ll = custom_gmm.log_likelihood_history_[-1]
    custom_bic = bic_score(custom_ll, n_params, X_train.shape[0])
    custom_aic = aic_score(custom_ll, n_params)
    custom_acc = clustering_accuracy(y_test, custom_test_preds)

    start_time = time.time()
    sklearn_gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=100,
        reg_covar=1e-6,
        random_state=RANDOM_STATE,
    )
    sklearn_gmm.fit(X_train)
    sklearn_time = time.time() - start_time

    sklearn_train_preds = sklearn_gmm.predict(X_train)
    sklearn_test_preds = sklearn_gmm.predict(X_test)
    sklearn_ll = sklearn_gmm.score(X_train) * len(X_train)
    sklearn_bic = sklearn_gmm.bic(X_train)
    sklearn_aic = sklearn_gmm.aic(X_train)
    sklearn_acc = clustering_accuracy(y_test, sklearn_test_preds)

    summary = pd.DataFrame(
        [
            {
                "model": "custom_gmm",
                "log_likelihood": custom_ll,
                "bic": custom_bic,
                "aic": custom_aic,
                "accuracy": custom_acc,
                "runtime_sec": custom_time,
            },
            {
                "model": "sklearn_gmm",
                "log_likelihood": sklearn_ll,
                "bic": sklearn_bic,
                "aic": sklearn_aic,
                "accuracy": sklearn_acc,
                "runtime_sec": sklearn_time,
            },
        ]
    )

    summary.to_csv(artifacts_dir / "summary.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(custom_gmm.log_likelihood_history_)
    plt.title("EM Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "em_convergence.png", dpi=180)
    plt.close()

    save_clusters_plot(
        X_train,
        custom_train_preds,
        artifacts_dir / "custom_gmm_clusters.png",
        "Custom GMM Clusters",
    )
    save_clusters_plot(
        X_train,
        sklearn_train_preds,
        artifacts_dir / "sklearn_gmm_clusters.png",
        "Sklearn GMM Clusters",
    )

    print("Custom GMM:")
    print(f"Log-Likelihood: {custom_ll:.4f}")
    print(f"BIC: {custom_bic:.4f}")
    print(f"AIC: {custom_aic:.4f}")
    print(f"Accuracy: {custom_acc:.4f}")
    print(f"Runtime: {custom_time:.4f} sec\n")

    print("Sklearn GMM:")
    print(f"Log-Likelihood: {sklearn_ll:.4f}")
    print(f"BIC: {sklearn_bic:.4f}")
    print(f"AIC: {sklearn_aic:.4f}")
    print(f"Accuracy: {sklearn_acc:.4f}")
    print(f"Runtime: {sklearn_time:.4f} sec")


if __name__ == "__main__":
    main()
