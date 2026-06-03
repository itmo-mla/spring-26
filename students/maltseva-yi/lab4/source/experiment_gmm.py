import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import os
import warnings
warnings.filterwarnings('ignore')

from gmm_custom import GaussianMixtureEM

def main():
    os.makedirs("images", exist_ok=True)

    # 1. Загрузка и подготовка данных
    print("Загрузка датасета Iris...")
    iris = load_iris()
    X = iris.data
    y_true = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Размерность данных: {X_scaled.shape}")
    print(f"Число компонент: 3 (предполагаем 3 вида ирисов)")

    # 2. Обучение собственной реализации
    print("\n--- Обучение собственного GMM (EM) ---")
    gmm_custom = GaussianMixtureEM(
        n_components=3,
        max_iter=300,
        tol=1e-4,
        reg_covar=1e-4,
        random_state=42
    )
    gmm_custom.fit(X_scaled)
    log_lik_custom = gmm_custom.score(X_scaled)
    print(f"Логарифм правдоподобия (собственная реализация): {log_lik_custom:.2f}")

    # 3. Обучение эталонной реализации sklearn
    print("\n--- Обучение sklearn GaussianMixture ---")
    gmm_sk = SklearnGMM(
        n_components=3,
        covariance_type='full',
        max_iter=300,
        tol=1e-4,
        reg_covar=1e-4,
        random_state=42
    )
    gmm_sk.fit(X_scaled)
    log_lik_sk = gmm_sk.score(X_scaled) * X_scaled.shape[0]
    print(f"Логарифм правдоподобия (sklearn): {log_lik_sk:.2f}")

    # 4. Сравнение
    print("\n=== Сравнение ===")
    print(f"Разница в log-likelihood: {abs(log_lik_custom - log_lik_sk):.3f}")

    pred_custom = gmm_custom.predict(X_scaled)
    pred_sk = gmm_sk.predict(X_scaled)
    ari = adjusted_rand_score(pred_custom, pred_sk)
    print(f"Adjusted Rand Index между разбиениями: {ari:.4f}")

    # 5. Визуализация (первые два признака)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=pred_custom, cmap='viridis', alpha=0.7)
    plt.title("Наша GMM (EM)")
    plt.xlabel("Признак 1 (стандартиз.)")
    plt.ylabel("Признак 2 (стандартиз.)")
    plt.colorbar(scatter1, label="Компонента")

    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=pred_sk, cmap='viridis', alpha=0.7)
    plt.title("sklearn GaussianMixture")
    plt.xlabel("Признак 1 (стандартиз.)")
    plt.ylabel("Признак 2 (стандартиз.)")
    plt.colorbar(scatter2, label="Компонента")

    plt.tight_layout()
    plt.savefig('images/gmm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # График сходимости EM
    plt.figure(figsize=(8, 5))
    plt.plot(gmm_custom.log_likelihood_, label='EM log-likelihood')
    plt.xlabel('Итерация')
    plt.ylabel('Log-likelihood')
    plt.title('Сходимость EM-алгоритма (собственная реализация)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('images/gmm_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 6. Вывод параметров (средних) для справки
    print("\n=== Средние компонент (обратное масштабирование) ===")
    print("Собственная реализация:")
    for j in range(3):
        mean_orig = scaler.inverse_transform(gmm_custom.means_[j].reshape(1, -1)).flatten()
        print(f"  Компонента {j}: {mean_orig[:2]} ...")
    print("\nsklearn:")
    for j in range(3):
        mean_orig = scaler.inverse_transform(gmm_sk.means_[j].reshape(1, -1)).flatten()
        print(f"  Компонента {j}: {mean_orig[:2]} ...")

if __name__ == "__main__":
    main()