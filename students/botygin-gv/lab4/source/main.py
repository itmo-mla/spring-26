import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.metrics import adjusted_rand_score

from gmm_model import GaussianMixtureEM


def main():
    X, y_true = make_blobs(
        n_samples=800,
        centers=3,
        n_features=2,
        cluster_std=[1.0, 2.5, 5.0],
        random_state=42
    )

    n_components = 3

    custom_gmm = GaussianMixtureEM(
        n_components=n_components,
        max_iter=200,
        tol=1e-5,
        reg_covar=1e-5,
        random_state=42
    )
    custom_gmm.fit(X)

    sklearn_gmm = SklearnGMM(
        n_components=n_components,
        random_state=43,
        tol=1e-5,
        max_iter=200,
        covariance_type='full'
    )
    sklearn_gmm.fit(X)

    custom_ll = custom_gmm.score(X)
    sklearn_ll = sklearn_gmm.score(X)

    custom_pred = custom_gmm.predict(X)
    sklearn_pred = sklearn_gmm.predict(X)

    custom_ari = adjusted_rand_score(y_true, custom_pred)
    sklearn_ari = adjusted_rand_score(y_true, sklearn_pred)

    print(f"Логарифм правдоподобия (ПМП) на 1 объект:")
    print(f"Custom EM : {custom_ll:.6f}")
    print(f"Sklearn   : {sklearn_ll:.6f}")
    print(f"Разница   : {abs(custom_ll - sklearn_ll):.6f}")
    print(f"Качество кластеризации (Adjusted Rand Index):")
    print(f"Custom EM : {custom_ari:.4f}")
    print(f"Sklearn   : {sklearn_ari:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', edgecolor='k', alpha=0.7)
    axes[0].set_title('Истинные метки (Ground Truth)')
    axes[0].axis('off')

    axes[1].scatter(X[:, 0], X[:, 1], c=custom_gmm.predict(X), cmap='plasma', edgecolor='k', alpha=0.7)
    axes[1].set_title('Custom GMM (EM)')
    axes[1].axis('off')

    axes[2].scatter(X[:, 0], X[:, 1], c=sklearn_gmm.predict(X), cmap='coolwarm', edgecolor='k', alpha=0.7)
    axes[2].set_title('Sklearn GaussianMixture')
    axes[2].axis('off')

    plt.tight_layout()
    fig.savefig("images/gmm_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()