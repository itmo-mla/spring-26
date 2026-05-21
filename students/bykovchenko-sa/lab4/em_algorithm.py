import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import multivariate_normal
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings
warnings.filterwarnings("ignore")
from itertools import permutations


def header(text: str):
    print(f"\n{'=' * 72}")
    print(f"  {text}")
    print(f"{'=' * 72}\n")


header("Датасет")
wine = datasets.load_wine()
X_raw = wine.data
y_true = wine.target

print(f"Размерность данных: X.shape = {X_raw.shape}")
print(f"Классов в данных: {len(np.unique(y_true))} (метки: {np.unique(y_true)})")
print(f"Признаки:")
for i, name in enumerate(wine.feature_names):
    print(f"    [{i:2d}] {name}")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print(f"\nПосле стандартизации: μ = {X.mean(axis=0).round(3)}")
print(f"Дисперсии: σ² = {X.var(axis=0).round(3)}")
print(f"PCA (2 компоненты): объяснённая дисперсия = {pca.explained_variance_ratio_.sum():.2%}")


def logsumexp(a, axis=None, keepdims=False):
    """
    Вычисляет log(Σ exp(a_i)) численно устойчивым способом
    logsumexp(a) = max(a) + log(Σ exp(a_i - max(a)))
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        if not keepdims:
            return result.squeeze(axis=axis)
        return result
    if not keepdims:
        return result.squeeze()
    return result


def multivariate_gaussian_pdf(X, mu, sigma):
    """
    Вычисляет значение многомерной плотности нормального распределения
    """
    d = X.shape[1]
    # добавляем маленькое значение к диагонали для избежания вырожденной ковариационной матрицы
    sigma_reg = sigma + 1e-4 * np.eye(d)
    return multivariate_normal.pdf(X, mean=mu, cov=sigma_reg)


class MyGMM:
    """
    Реализация Gaussian Mixture Model (GMM) методом EM-алгоритма
    """

    def __init__(self, n_components=3, max_iter=200, tol=1e-6, n_init=5, random_state=42):
        self.n_components = n_components  # количество компонент смеси (гауссиан)
        self.max_iter = max_iter  # максимальное число итераций EM
        self.tol = tol  # порог сходимости по логарифму правдоподобия
        self.n_init = n_init   # количество независимых запусков с разными начальными значениями
        self.random_state = random_state

        # параметры модели
        self.weights_ = None      # π_k — априорные вероятности компонент
        self.means_ = None        # μ_k — векторы средних
        self.covariances_ = None  # Σ_k — ковариационные матрицы
        self.labels_ = None       # индексы компонент для каждого образца
        self.log_likelihood_history_ = []  # история log-likelihood
        self.converged_iter_ = None
        self.best_log_likelihood_ = -np.inf

    def _initialize_params(self, X, rng):
        """
        Инициализация параметров модели через KMeans
        Инициализация EM определяет, к какому локальному максимуму сойдётся алгоритм
        """
        n, d = X.shape
        K = self.n_components

        kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=rng.integers(0, 100000))
        kmeans_labels = kmeans.fit_predict(X)

        # априорные веса: пропорциональны размеру кластера KMeans
        weights = np.array([np.sum(kmeans_labels == k) / n for k in range(K)])

        # средние: центроиды кластеров KMeans
        means = kmeans.cluster_centers_.copy()

        # ковариации: выборочная ковариация внутри каждого кластера.
        covariances = np.zeros((K, d, d))
        for k in range(K):
            mask = kmeans_labels == k
            if np.sum(mask) > d + 1:
                covariances[k] = np.cov(X[mask], rowvar=False)
            else:
                # если кластер слишком мал — используем общую ковариацию
                covariances[k] = np.cov(X, rowvar=False)
            if covariances[k].ndim == 0:
                covariances[k] = np.array([[covariances[k]]])
            # регуляризация для численной устойчивости
            covariances[k] += 1e-4 * np.eye(d)

        return weights, means, covariances

    def _e_step(self, X, weights, means, covariances):
        """
        E-шаг (Expectation step).
        На E-шаге вычисляются апостериорные вероятности принадлежности каждого образца каждой компоненте смеси
        """
        n = X.shape[0]
        K = self.n_components

        # вычисляем логарифмы компонент π_k · N(x_i | μ_k, Σ_k)
        # для численной устойчивости работаем в логарифмическом масштабе log(π_k · N) = log π_k + log N(x|μ_k,Σ_k)
        log_resp = np.zeros((n, K))
        for k in range(K):
            log_resp[:, k] = (np.log(weights[k] + 1e-300) +
                              multivariate_normal.logpdf(X, mean=means[k], cov=covariances[k] + 1e-4 * np.eye(X.shape[1])))

        log_likelihood_total = np.sum(logsumexp(log_resp, axis=1))
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        responsibilities = np.exp(log_resp - log_norm)

        return responsibilities, log_likelihood_total

    def _m_step(self, X, responsibilities):
        """
        M-шаг (Maximization step)
        На M-шаге параметры θ пересчитываются по принципу взвешенного максимума правдоподобия (Weighted MLE)
        """
        n, d = X.shape
        K = self.n_components

        # эффективные размеры компонент
        N_k = np.sum(responsibilities, axis=0)  # (K,)

        # предотвращаем деление на 0 для пустых компонент
        N_k = np.maximum(N_k, 1e-10)

        # обновление весов π_k = N_k / N
        weights = N_k / n

        # обновление средних μ_k = (1/N_k) Σ_i γ(z_{ik}) · x_i
        means = np.zeros((K, d))
        for k in range(K):
            means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]

        # обновление ковариаций Σ_k = (1/N_k) Σ_i γ(z_{ik}) · (x_i - μ_k)(x_i - μ_k)ᵀ
        covariances = np.zeros((K, d, d))
        for k in range(K):
            diff = X - means[k]  # (n, d)
            # взвешенная сумма внешних произведений
            covariances[k] = (np.dot((responsibilities[:, k:k+1] * diff).T, diff) / N_k[k])
            covariances[k] += 1e-4 * np.eye(d)

        return weights, means, covariances

    def fit(self, X):
        n, d = X.shape
        K = self.n_components

        best_log_likelihood = -np.inf
        best_params = None

        for init in range(self.n_init):
            # каждый запуск EM с различной инициализацией может сойтись к разному локальному максимуму, сохраняем лучший
            rng = np.random.default_rng(self.random_state + init)
            weights, means, covariances = self._initialize_params(X, rng)
            ll_history = []

            header(f"   EM запуск {init + 1}/{self.n_init}")
            for iteration in range(self.max_iter):
                # E-шаг. Вычисляем апостериорные вероятности по формуле Байеса
                responsibilities, log_ll = self._e_step(X, weights, means, covariances)
                ll_history.append(log_ll)

                # M-шаг. Обновляем параметры по взвешенному МП
                weights, means, covariances = self._m_step(X, responsibilities)

                # проверка сходимости
                if iteration > 0:
                    delta = abs(ll_history[-1] - ll_history[-2])
                    if delta < self.tol:
                        print(f"    Сходится на итерации {iteration + 1}: log L = {ll_history[-1]:.4f}, Δ = {delta:.2e}")
                        break

            final_ll = ll_history[-1]
            print(f"    Итоговый log-likelihood: {final_ll:.4f} ({iteration + 1} итераций)")

            # сохраняем лучший запуск
            if final_ll > best_log_likelihood:
                best_log_likelihood = final_ll
                best_params = {
                    "weights": weights.copy(),
                    "means": means.copy(),
                    "covariances": covariances.copy(),
                    "responsibilities": responsibilities.copy(),
                    "ll_history": ll_history.copy(),
                    "converged_iter": iteration + 1,
                }

        # применяем лучшие параметры
        self.weights_ = best_params["weights"]
        self.means_ = best_params["means"]
        self.covariances_ = best_params["covariances"]
        self.labels_ = np.argmax(best_params["responsibilities"], axis=1)
        self.log_likelihood_history_ = best_params["ll_history"]
        self.converged_iter_ = best_params["converged_iter"]
        self.best_log_likelihood_ = best_log_likelihood

        print(f"\n  >>> Лучший log-likelihood: {self.best_log_likelihood_:.4f}")
        return self

    def predict(self, X):
        responsibilities, _ = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """Возвращает матрицу апостериорных вероятностей (n, K)"""
        responsibilities, _ = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return responsibilities

    def score(self, X):
        """
        Вычисляет логарифм полного правдоподобия (log-likelihood)
        """
        _, log_ll = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return log_ll

    def compute_aic_bic(self, X):
        n, d = X.shape
        K = self.n_components

        # число свободных параметров
        n_params = (K * d +                    # средние
                    K * d * (d + 1) // 2 +     # ковариации (симметричные)
                    (K - 1))                   # веса (Σπ_k = 1 → K-1 свободных)

        log_ll = self.score(X)
        aic = -2 * log_ll + 2 * n_params
        bic = -2 * log_ll + n_params * np.log(n)
        return aic, bic


header("Обучаем свою GMM (EM-алгоритм)")
K = 3
gmm_custom = MyGMM(n_components=K, max_iter=300, tol=1e-6, n_init=10, random_state=42)
gmm_custom.fit(X)

print(f"\nОбученные параметры (from-scratch GMM):")
print(f"  Количество компонент K = {K}")
print(f"  Априорные веса π_k: {gmm_custom.weights_.round(4)}")
print(f"  Векторы средних μ_k:\n{gmm_custom.means_.round(4)}")
print(f"  Итераций до сходимости: {gmm_custom.converged_iter_}")

header("Обучаем эталонную GMM (scikit-learn)")
gmm_sklearn = GaussianMixture(n_components=K, max_iter=300, tol=1e-6, n_init=10, random_state=42, covariance_type='full')
gmm_sklearn.fit(X)
labels_sklearn = gmm_sklearn.predict(X)

print(f"sklearn GMM обучена.")
print(f"  Веса: {gmm_sklearn.weights_.round(4)}")
print(f"  Средние:\n{gmm_sklearn.means_.round(4)}")
print(f"  Конвергенция: iter = {gmm_sklearn.n_iter_}")


header("Оценка качества")
ll_custom = gmm_custom.score(X)
ll_sklearn_total = gmm_sklearn.score(X) * X.shape[0]
ll_sklearn_avg = gmm_sklearn.score(X)

print(f"Log-likelihood (моя реализация, полная сумма): {ll_custom:.4f}")
print(f"Log-likelihood (sklearn, полная сумма):        {ll_sklearn_total:.4f}")
print(f"Log-likelihood (sklearn, средний):             {ll_sklearn_avg:.4f}")
print(f"Разница (полных сумм):                         {abs(ll_custom - ll_sklearn_total):.4f}")


aic_custom, bic_custom = gmm_custom.compute_aic_bic(X)
aic_sklearn = gmm_sklearn.aic(X)
bic_sklearn = gmm_sklearn.bic(X)

print(f"\nAIC (моя):    {aic_custom:.4f}")
print(f"AIC (sklearn): {aic_sklearn:.4f}")
print(f"BIC (моя):    {bic_custom:.4f}")
print(f"BIC (sklearn): {bic_sklearn:.4f}")


header("Сравнение точности моделей")
labels_custom = gmm_custom.labels_

ari_custom = adjusted_rand_score(y_true, labels_custom)
ari_sklearn = adjusted_rand_score(y_true, labels_sklearn)

print(f"Adjusted Rand Index (ARI) относительно истинных меток:")
print(f"  Моя GMM:     {ari_custom:.4f}")
print(f"  sklearn GMM:  {ari_sklearn:.4f}")
print(f"  (ARI=1.0 — идеальное совпадение, ARI≈0 — случайное)")

sil_custom = silhouette_score(X, labels_custom)
sil_sklearn = silhouette_score(X, labels_sklearn)

print(f"\nSilhouette Score:")
print(f"  Моя GMM:     {sil_custom:.4f}")
print(f"  sklearn GMM:  {sil_sklearn:.4f}")
print(f"  (близко к 1.0 — компактные, хорошо разделённые кластеры)")

resp_custom = gmm_custom.predict_proba(X)
resp_sklearn = gmm_sklearn.predict_proba(X)

best_resp_diff = np.inf

for perm in permutations(range(K)):
    diff = np.mean((resp_custom - resp_sklearn[:, perm]) ** 2)
    if diff < best_resp_diff:
        best_resp_diff = diff

print(f"\nУсреднённая разность матриц ответственности (MSE): {best_resp_diff:.6f}")
print(f"  (с учётом оптимальной перестановки компонент)")

header("Выбор оптимального числа компонента K")
K_range = range(1, 8)
bic_values_custom = []
bic_values_sklearn = []
ll_values_custom = []

for k in K_range:
    gmm_k = MyGMM(n_components=k, max_iter=300, tol=1e-6, n_init=3, random_state=42)
    gmm_k.fit(X)
    _, bic_k = gmm_k.compute_aic_bic(X)
    bic_values_custom.append(bic_k)
    ll_values_custom.append(gmm_k.score(X))
    gmm_s = GaussianMixture(n_components=k, max_iter=300, tol=1e-6, n_init=3, random_state=42)
    gmm_s.fit(X)
    bic_values_sklearn.append(gmm_s.bic(X))

print(f"{'K':>4} | {'BIC (моя)':>14} | {'BIC (sklearn)':>14} | {'Log-LL (моя)':>16}")
print("-" * 56)
for i, k in enumerate(K_range):
    print(f"{k:>4} | {bic_values_custom[i]:>14.2f} | {bic_values_sklearn[i]:>14.2f} "
          f"| {ll_values_custom[i]:>16.2f}")

best_k_bic = list(K_range)[np.argmin(bic_values_custom)]
print(f"\nОптимальное K по BIC (моя модель): {best_k_bic}")


OUT = "plots"

# сходимость EM-алгоритма
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(gmm_custom.log_likelihood_history_, 'b-o', markersize=3, label='Log-likelihood')
ax.set_xlabel("Итерация")
ax.set_ylabel("L(θ) = Σ log p(x|θ)")
ax.set_title("Сходимость EM-алгоритма")
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/lab4_plot_convergence.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# кластеризация
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_custom, cmap='viridis', s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Моя GMM (K=3)")
plt.colorbar(scatter, ax=ax, label="Компонента")

ax = axes[1]
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_sklearn, cmap='viridis', s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("sklearn GMM (K=3)")
plt.colorbar(scatter, ax=ax, label="Компонента")

fig.suptitle("Кластеризация PCA-проекции", fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{OUT}/lab4_plot_clustering.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# истинные классы
fig, ax = plt.subplots(figsize=(7, 5))
class_names = wine.target_names
colors_true = ['#ff7f0e', '#2ca02c', '#1f77b4']
for cls in range(3):
    mask = y_true == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors_true[cls], s=30, alpha=0.7, edgecolors='k', linewidths=0.3, label=class_names[cls])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Истинные классы")
ax.legend(loc='best')
plt.tight_layout()
fig.savefig(f"{OUT}/lab4_plot_true_classes.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# BIC vs K
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(list(K_range), bic_values_custom, 'bo-', label='BIC (моя)')
ax.plot(list(K_range), bic_values_sklearn, 'rs--', label='BIC (sklearn)')
ax.axvline(x=best_k_bic, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel("Число компонент K")
ax.set_ylabel("BIC")
ax.set_title("Выбор K по BIC")
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.annotate(
    f"Оптимальное K={best_k_bic}",
    xy=(best_k_bic, min(bic_values_custom)),
    xytext=(best_k_bic + 1.5, min(bic_values_custom) + 20),
    arrowprops=dict(arrowstyle='->', color='gray'),
    fontsize=9
)
plt.tight_layout()
fig.savefig(f"{OUT}/lab4_plot_bic.png", dpi=150, bbox_inches='tight')
plt.close(fig)
