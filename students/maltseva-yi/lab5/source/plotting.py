import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

def plot_rmse_comparison(rmse_own, rmse_slim_etalon, rmse_als_own, rmse_als_etalon):
    models = ['SLIM свой']
    values = [rmse_own]
    if rmse_slim_etalon is not None:
        models.append('SLIM эталон')
        values.append(rmse_slim_etalon)
    models.append('ALS свой')
    values.append(rmse_als_own)
    if rmse_als_etalon is not None:
        models.append('ALS эталон')
        values.append(rmse_als_etalon)

    plt.figure(figsize=(8,5))
    bars = plt.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    plt.ylabel('RMSE')
    plt.title('Сравнение моделей на MovieLens 100k')
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('images/rmse_comparison.png')
    plt.show()

def plot_als_convergence(train_rmse_hist):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_rmse_hist)+1), train_rmse_hist, marker='o')
    plt.xlabel('Итерация')
    plt.ylabel('Train RMSE')
    plt.title('Сходимость ALS (собственная реализация)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/als_convergence.png')
    plt.show()

def plot_als_pred_vs_true(y_true, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolors='none')
    plt.plot([1,5], [1,5], 'r--')
    plt.xlabel('Истинный рейтинг')
    plt.ylabel('Предсказанный рейтинг')
    plt.title('ALS: предсказания vs истина')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/als_pred_vs_true.png')
    plt.show()

def plot_als_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Ошибка (истина - предсказание)')
    plt.ylabel('Частота')
    plt.title('Распределение ошибок ALS')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('images/als_residuals.png')
    plt.show()

def plot_slim_sparsity(W_own):
    plt.figure(figsize=(6,6))
    plt.spy(csr_matrix(W_own), markersize=1, aspect='auto')
    plt.title('Разреженность матрицы W (SLIM)')
    plt.tight_layout()
    plt.savefig('images/slim_sparsity.png')
    plt.show()