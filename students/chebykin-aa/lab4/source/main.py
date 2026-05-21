import logging
import os
import shutil
import time

from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import GaussianMixture as CustomGMM
from utils import load_data, evaluate_gmm, plot_pca_clusters


def main():
    SEED = 42
    K = 3

    results_path = os.path.join(os.getcwd(), "students", "chebykin-aa", "lab4", "results")
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(results_path, "main.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    X, _ = load_data()
    logging.info(f"Датасет: {X.shape[0]} объектов, {X.shape[1]} признаков")

    X_train_raw, X_test_raw = train_test_split(X, test_size=0.20, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # обучение Custom GMM
    t0 = time.perf_counter()
    custom_gmm = CustomGMM(n_components=K, random_state=SEED)
    custom_gmm.fit(X_train)
    custom_time = time.perf_counter() - t0
    logging.info(f"Custom GMM: converged={custom_gmm.converged_}, n_iter={custom_gmm.n_iter_}, time={custom_time:.3f}s")

    # обучение sklearn GMM
    t0 = time.perf_counter()
    sk_gmm = SklearnGMM(n_components=K, covariance_type="diag", random_state=SEED, max_iter=200)
    sk_gmm.fit(X_train)
    sklearn_time = time.perf_counter() - t0
    logging.info(f"sklearn GMM: converged={sk_gmm.converged_}, n_iter={sk_gmm.n_iter_}, time={sklearn_time:.3f}s")

    # оценка через ПМП
    custom_m = evaluate_gmm(custom_gmm, X_train, X_test, "Custom GMM",
                             save_path=os.path.join(results_path, "custom_gmm_test.txt"))
    sklearn_m = evaluate_gmm(sk_gmm, X_train, X_test, "sklearn GMM",
                              save_path=os.path.join(results_path, "sklearn_gmm_test.txt"))

    with open(os.path.join(results_path, "comparison.txt"), "w", encoding="utf-8") as f:
        f.write(f"K = {K}\n\n")
        f.write(f"Время обучения Custom GMM : {custom_time:.3f}s\n")
        f.write(f"Время обучения sklearn GMM: {sklearn_time:.3f}s\n")
        f.write(f"Соотношение (custom/sklearn): {custom_time / sklearn_time:.1f}x\n\n")
        f.write(f"Custom GMM:  LL(train)={custom_m['train_ll']:.4f}, LL(test)={custom_m['test_ll']:.4f}, BIC={custom_m['bic']:.2f}, AIC={custom_m['aic']:.2f}\n")
        f.write(f"sklearn GMM: LL(train)={sklearn_m['train_ll']:.4f}, LL(test)={sklearn_m['test_ll']:.4f}, BIC={sklearn_m['bic']:.2f}, AIC={sklearn_m['aic']:.2f}\n")

    with open(os.path.join(results_path, "component_weights.txt"), "w", encoding="utf-8") as f:
        f.write("Веса компонент:\n")
        for k in range(K):
            f.write(f"  {k}: custom={custom_gmm.weights_[k]:.4f}, sklearn={sk_gmm.weights_[k]:.4f}\n")

    plot_pca_clusters(X_test, custom_gmm.predict(X_test), sk_gmm.predict(X_test),
                      os.path.join(results_path, "pca_clusters.png"))

    logging.info("Done.")
    print("Pipeline завершён. Результаты сохранены в", results_path)


if __name__ == "__main__":
    main()
