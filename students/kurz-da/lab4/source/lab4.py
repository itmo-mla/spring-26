import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import time

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ============================================================
#  GMM (собственная реализация EM-алгоритма)
# ============================================================

class GMMCustom:
    def __init__(self, n_components=3, max_iter=100, tol=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _initialize(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # веса равномерно
        self.weights = np.full(self.n_components, 1.0 / self.n_components)

        # средние - случайные точки из данных
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices].copy()

        # ковариации - единичные матрицы
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])

    def _multivariate_gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        diff = X - mean
        # регуляризация для численной стабильности
        cov_reg = cov + 1e-6 * np.eye(n_features)

        try:
            L = np.linalg.cholesky(cov_reg)
            # log det = 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            # solve L * z = diff^T
            z = np.linalg.solve(L, diff.T)
            mahal = np.sum(z ** 2, axis=0)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov_reg)
            log_det = np.log(np.abs(np.linalg.det(cov_reg)) + 1e-300)
            mahal = np.sum(diff @ inv_cov * diff, axis=1)

        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)
        return np.exp(log_prob)

    def _e_step(self, X):
        # E-шаг: считаю ответственности
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k]
            )

        total = responsibilities.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1e-300)
        responsibilities /= total

        return responsibilities

    def _m_step(self, X, responsibilities):
        # M-шаг: обновляю параметры
        n_samples, n_features = X.shape

        for k in range(self.n_components):
            resp_k = responsibilities[:, k]
            N_k = resp_k.sum()

            if N_k < 1e-10:
                continue

            self.means[k] = (resp_k[:, np.newaxis] * X).sum(axis=0) / N_k

            diff = X - self.means[k]
            self.covariances[k] = (resp_k[:, np.newaxis] * diff).T @ diff / N_k

            self.weights[k] = N_k / n_samples

    def _log_likelihood(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            likelihood += self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k]
            )

        return np.sum(np.log(likelihood + 1e-300))

    def fit(self, X):
        self._initialize(X)
        prev_ll = -np.inf
        self.log_likelihoods = []
        self.n_iter_ = 0

        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            ll = self._log_likelihood(X)
            self.log_likelihoods.append(ll)
            self.n_iter_ = iteration + 1

            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def score(self, X):
        # средний log-likelihood на объект
        return self._log_likelihood(X) / X.shape[0]

    def bic(self, X):
        n_samples, n_features = X.shape
        # число параметров: веса + средние + ковариации
        n_params = (self.n_components - 1)  # веса (K-1 из-за нормировки)
        n_params += self.n_components * n_features  # средние
        n_params += self.n_components * n_features * (n_features + 1) // 2  # ковариации (симм.)
        ll = self._log_likelihood(X)
        return -2 * ll + n_params * np.log(n_samples)


# ============================================================
#  Основной скрипт
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  Лабораторная работа №4. EM-алгоритм")
    print("  Gaussian Mixture Model")
    print("=" * 60)

    # загружаю данные
    iris = load_iris()
    X_raw = iris.data
    y_true = iris.target

    # стандартизирую
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"\nДатасет: Iris")
    print(f"Размер: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print(f"Классы: {np.unique(y_true)}, по {np.bincount(y_true)} объектов")

    # обучаю GMM
    print("\n-- Обучение моей GMM (n_components=3) --")
    t0 = time.time()
    my_gmm = GMMCustom(n_components=3, max_iter=200, tol=1e-6, random_state=42)
    my_gmm.fit(X)
    my_time = time.time() - t0

    my_labels = my_gmm.predict(X)
    my_ll = my_gmm.score(X)
    my_ari = adjusted_rand_score(y_true, my_labels)

    print(f"Итераций до сходимости: {my_gmm.n_iter_}")
    print(f"Log-likelihood (ср.):   {my_ll:.4f}")
    print(f"ARI:                    {my_ari:.4f}")
    print(f"Время обучения:         {my_time:.4f} сек")

    # сходимость
    print("\nСходимость log-likelihood:")
    steps_to_show = min(10, len(my_gmm.log_likelihoods))
    step_size = max(1, len(my_gmm.log_likelihoods) // steps_to_show)
    for i in range(0, len(my_gmm.log_likelihoods), step_size):
        print(f"  Итерация {i+1:>3}: {my_gmm.log_likelihoods[i]:.4f}")
    if (len(my_gmm.log_likelihoods) - 1) % step_size != 0:
        print(f"  Итерация {len(my_gmm.log_likelihoods):>3}: {my_gmm.log_likelihoods[-1]:.4f}")

    # sklearn для сравнения
    print("\n-- Эталон: sklearn GaussianMixture --")
    t0 = time.time()
    sk_gmm = GaussianMixture(n_components=3, max_iter=200, tol=1e-6, random_state=42)
    sk_gmm.fit(X)
    sk_time = time.time() - t0

    sk_labels = sk_gmm.predict(X)
    sk_ll = sk_gmm.score(X)
    sk_ari = adjusted_rand_score(y_true, sk_labels)

    print(f"Итераций до сходимости: {sk_gmm.n_iter_}")
    print(f"Log-likelihood (ср.):   {sk_ll:.4f}")
    print(f"ARI:                    {sk_ari:.4f}")
    print(f"Время обучения:         {sk_time:.4f} сек")

    # подбор числа компонент
    print("\n-- Подбор числа компонент (BIC) --")
    print(f"{'K':<5} {'BIC (мой)':<18} {'BIC (sklearn)':<18} {'ARI (мой)':<12} {'ARI (sklearn)':<12}")
    print("-" * 65)
    k_grid = [2, 3, 4, 5, 6]
    my_bics, sk_bics, my_aris, sk_aris = [], [], [], []
    for k in k_grid:
        my_g = GMMCustom(n_components=k, max_iter=200, random_state=42)
        my_g.fit(X)
        my_bic_val = my_g.bic(X)
        my_ari_k = adjusted_rand_score(y_true, my_g.predict(X))

        sk_g = GaussianMixture(n_components=k, max_iter=200, random_state=42)
        sk_g.fit(X)
        sk_bic_val = sk_g.bic(X)
        sk_ari_k = adjusted_rand_score(y_true, sk_g.predict(X))

        my_bics.append(my_bic_val)
        sk_bics.append(sk_bic_val)
        my_aris.append(my_ari_k)
        sk_aris.append(sk_ari_k)

        print(f"{k:<5} {my_bic_val:<18.4f} {sk_bic_val:<18.4f} {my_ari_k:<12.4f} {sk_ari_k:<12.4f}")

    # итоговая таблица
    print("\n" + "=" * 60)
    print("  Сравнение результатов (K=3)")
    print("=" * 60)
    print(f"{'Метрика':<30} {'Моя реализация':<20} {'sklearn':<15}")
    print("-" * 65)
    print(f"{'Log-likelihood (ср.)':<30} {my_ll:<20.4f} {sk_ll:<15.4f}")
    print(f"{'ARI':<30} {my_ari:<20.4f} {sk_ari:<15.4f}")
    print(f"{'Итераций':<30} {my_gmm.n_iter_:<20} {sk_gmm.n_iter_:<15}")
    print(f"{'BIC':<30} {my_gmm.bic(X):<20.4f} {sk_gmm.bic(X):<15.4f}")
    print(f"{'Время обучения (сек)':<30} {my_time:<20.4f} {sk_time:<15.4f}")

    # ============================================================
    #  Графики
    # ============================================================

    # 1. сходимость log-likelihood
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(my_gmm.log_likelihoods) + 1), my_gmm.log_likelihoods,
            marker='o', color='#1f77b4', linewidth=2, markersize=4)
    ax.set_xlabel('Итерация EM')
    ax.set_ylabel('Log-likelihood')
    ax.set_title('Сходимость EM-алгоритма (n_components=3, Iris)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'em_convergence.png'), dpi=100)
    plt.close()

    # 2. BIC vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_grid, my_bics, marker='o', label='Моя реализация',
            color='#1f77b4', linewidth=2)
    ax.plot(k_grid, sk_bics, marker='s', label='sklearn',
            color='#ff7f0e', linewidth=2)
    ax.set_xlabel('Число компонент K')
    ax.set_ylabel('BIC')
    ax.set_title('Подбор числа компонент по BIC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'bic_vs_k.png'), dpi=100)
    plt.close()

    # 3. кластеризация в 2D (PCA)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Истинные классы', 'Моя GMM', 'sklearn GMM']
    label_sets = [y_true, my_labels, sk_labels]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for ax, title, labels in zip(axes, titles, label_sets):
        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=colors[cls % len(colors)], label=f'Класс {cls}',
                       alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'clusters_2d.png'), dpi=100)
    plt.close()

    # 4. ARI vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_grid, my_aris, marker='o', label='Моя реализация',
            color='#1f77b4', linewidth=2)
    ax.plot(k_grid, sk_aris, marker='s', label='sklearn',
            color='#ff7f0e', linewidth=2)
    ax.set_xlabel('Число компонент K')
    ax.set_ylabel('ARI')
    ax.set_title('Качество кластеризации (ARI) vs число компонент')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'ari_vs_k.png'), dpi=100)
    plt.close()

    print(f"\nГрафики сохранены в {IMAGES_DIR}")
