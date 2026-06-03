import os
from sklearn.mixture import GaussianMixture as SklearnGMM

from gmm import GMM
from data_loader import load_wine_dataset, train_test_split_manual
from visualization import (
    plot_clusters,
    plot_log_likelihood,
    plot_log_likelihood_comparison,
)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
N_COMPONENTS_RANGE = [2, 3, 4, 5, 6, 7]
BEST_K = 3



def run_comparison(X_train, X_test):
    scores_custom = []
    scores_sklearn = []

    for k in N_COMPONENTS_RANGE:
        print(f"\n  k = {k}")

        custom_model = GMM(n_components=k, max_iter=200, tol=1e-4,
                           random_state=RANDOM_STATE)
        custom_model.fit(X_train)
        score_c = custom_model.score(X_test)
        scores_custom.append(score_c)

        sklearn_model = SklearnGMM(n_components=k, max_iter=200, tol=1e-4,
                                   random_state=RANDOM_STATE,
                                   covariance_type="full")
        sklearn_model.fit(X_train)
        score_s = sklearn_model.score(X_test)
        scores_sklearn.append(score_s)

        print(f"    Собственная: {score_c:.4f}  |  sklearn: {score_s:.4f}  "
              f"|  разница: {abs(score_c - score_s):.4f}")

    return scores_custom, scores_sklearn


def print_summary_table(scores_custom, scores_sklearn):
    print("\n" + "=" * 56)
    print(f"{'k':>4} | {'Собственная':>14} | {'sklearn':>14} | {'|Δ|':>8}")
    print("-" * 56)
    for i, k in enumerate(N_COMPONENTS_RANGE):
        diff = abs(scores_custom[i] - scores_sklearn[i])
        print(f"{k:>4} | {scores_custom[i]:>14.4f} | "
              f"{scores_sklearn[i]:>14.4f} | {diff:>8.4f}")
    print("=" * 56)



def run_convergence(X_train):
    model = GMM(n_components=BEST_K, max_iter=200, tol=1e-6,
                random_state=RANDOM_STATE)
    model.fit(X_train)
    print(f"  Сошёлся за {len(model.log_likelihood_history_)} итераций")
    return model


def run_cluster_visualization(X_train):
    X_2d = X_train[:, :2]

    custom_model = GMM(n_components=BEST_K, max_iter=200, tol=1e-4,
                       random_state=RANDOM_STATE)
    custom_model.fit(X_2d)
    labels_c = custom_model.predict(X_2d)

    plot_clusters(
        X_2d, labels_c,
        custom_model.means_, custom_model.covariances_,
        title=f"GMM — собственная реализация (k={BEST_K})",
        filename=os.path.join(PLOTS_DIR, "clusters_custom.png"),
    )

    sklearn_model = SklearnGMM(n_components=BEST_K, max_iter=200,
                                random_state=RANDOM_STATE,
                                covariance_type="full")
    sklearn_model.fit(X_2d)
    labels_s = sklearn_model.predict(X_2d)

    plot_clusters(
        X_2d, labels_s,
        sklearn_model.means_, sklearn_model.covariances_,
        title=f"GMM — sklearn (k={BEST_K})",
        filename=os.path.join(PLOTS_DIR, "clusters_sklearn.png"),
    )


def main():
    X, feature_names = load_wine_dataset()
    X_train, X_test = train_test_split_manual(X, test_size=0.2,
                                               random_state=RANDOM_STATE)
    print(f"Train: {X_train.shape},  Test: {X_test.shape}")

    print("\nСравнение при разных k...")
    scores_c, scores_s = run_comparison(X_train, X_test)
    print_summary_table(scores_c, scores_s)

    best_model = run_convergence(X_train)
    plot_log_likelihood(
        best_model.log_likelihood_history_,
        k=BEST_K,
        filename=os.path.join(PLOTS_DIR, "convergence.png"),
    )

    plot_log_likelihood_comparison(
        N_COMPONENTS_RANGE, scores_c, scores_s,
        filename=os.path.join(PLOTS_DIR, "score_comparison.png"),
    )

    run_cluster_visualization(X_train)

    print(f"\nГрафики сохранены в: {PLOTS_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()