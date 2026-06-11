import numpy as np
from sklearn.mixture import GaussianMixture as SklearnGMM

from src.data import load_data
from src.gmm import GaussianMixture
from src.evaluation import bic, aic
import src.plots as plots

N_COMPONENTS = 5


def main():
    print("Lab 4")

    print("\nLoading dataset...")
    X, name = load_data()
    print(f"Shape : {X.shape}")
    plots.plot_data(X, name)

    print("\nModel selection (k = 2..8)...")
    ks = list(range(2, 9))
    bics, aics = [], []
    for k in ks:
        m = GaussianMixture(n_components=k, random_state=42)
        m.fit(X)
        b, a = bic(m, X), aic(m, X)
        bics.append(b)
        aics.append(a)
        print(f"k={k}  BIC={b:9.2f}  AIC={a:9.2f}")
    plots.plot_model_selection(ks, bics, aics)
    best_k = ks[int(np.argmin(bics))]
    print(f"\nBest k by BIC: {best_k}")

    print(f"\nTraining Custom GMM  (k={N_COMPONENTS})...")
    custom = GaussianMixture(n_components=N_COMPONENTS, random_state=42)
    custom.fit(X)
    custom_ll = custom.score(X)
    print(f"Log-likelihood/sample: {custom_ll:.4f}")
    print(f"Iterations: {len(custom.log_likelihoods_)}")
    plots.plot_gmm(X, custom, f"Custom GMM  (k={N_COMPONENTS})", "gmm_custom.png")
    plots.plot_convergence(custom.log_likelihoods_)

    print(f"\nTraining Sklearn GMM (k={N_COMPONENTS})...")
    sklearn = SklearnGMM(n_components=N_COMPONENTS, random_state=42,
                         max_iter=200, n_init=1)
    sklearn.fit(X)
    sklearn_ll = sklearn.score(X)
    print(f"Log-likelihood/sample: {sklearn_ll:.4f}")
    print(f"Iterations: {sklearn.n_iter_}")
    plots.plot_gmm(X, sklearn, f"Sklearn GMM  (k={N_COMPONENTS})", "gmm_sklearn.png")

    print("\nComparison:")
    print(f"{'Model':<22} {'LL/sample':>12}  {'BIC':>10}")
    print(f"{'-'*46}")
    custom_bic  = bic(custom, X)
    sklearn_bic = bic(sklearn, X)
    print(f"{'Custom GMM':<22} {custom_ll:>12.4f}  {custom_bic:>10.2f}")
    print(f"{'Sklearn GMM':<22} {sklearn_ll:>12.4f}  {sklearn_bic:>10.2f}")
    print(f"\nLL  (custom - sklearn): {custom_ll - sklearn_ll:+.4f}")
    print(f"BIC (custom - sklearn): {custom_bic - sklearn_bic:+.2f}")


if __name__ == "__main__":
    main()
