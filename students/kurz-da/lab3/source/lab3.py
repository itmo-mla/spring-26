import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
import time

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ============================================================
#  Градиентный бустинг (собственная реализация)
# ============================================================

class GradientBoostingCustom:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_leaf=5, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _negative_gradient(self, y, pred):
        # антиградиент log loss
        return y - self._sigmoid(pred)

    def fit(self, X, y, track_loss=False):
        np.random.seed(self.random_state)
        self.trees = []
        self.loss_history = []

        # начальное приближение: log(p / (1-p))
        p = np.mean(y)
        self.initial_prediction = np.log((p + 1e-15) / (1.0 - p + 1e-15))
        pred = np.full(len(y), self.initial_prediction)

        for i in range(self.n_estimators):
            residuals = self._negative_gradient(y, pred)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X, residuals)

            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

            if track_loss:
                proba = self._sigmoid(pred)
                proba = np.clip(proba, 1e-15, 1 - 1e-15)
                loss = -np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba))
                self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ============================================================
#  Кросс-валидация (ручная реализация)
# ============================================================

def cross_validate_manual(model_class, X, y, params, n_folds=5, random_state=42):
    np.random.seed(random_state)
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    scores = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

        model = model_class(**params)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        scores.append(accuracy_score(y[val_idx], pred))

    return np.array(scores)


# ============================================================
#  Основной скрипт
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  Лабораторная работа №3. Градиентный бустинг")
    print("=" * 60)

    # загружаю данные
    data = load_breast_cancer()
    X, y = data.data, data.target

    print(f"\nДатасет: Breast Cancer Wisconsin")
    print(f"Размер: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print(f"Классы: {np.unique(y)}, баланс: {np.bincount(y)}")

    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_leaf': 5,
        'random_state': 42
    }

    # кросс-валидация моей реализации
    print("\n-- Кросс-валидация (5 фолдов) --")
    print("\nМоя реализация:")
    t0 = time.time()
    my_scores = cross_validate_manual(GradientBoostingCustom, X, y, params, n_folds=5)
    my_cv_time = time.time() - t0

    print(f"Accuracy по фолдам: {my_scores}")
    print(f"Среднее accuracy:   {np.mean(my_scores):.4f} (+/- {np.std(my_scores):.4f})")
    print(f"Время:              {my_cv_time:.4f} сек")

    # кросс-валидация sklearn
    print("\nЭталон sklearn GradientBoostingClassifier:")
    sk_model = GradientBoostingClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    t0 = time.time()
    sk_scores = cross_val_score(sk_model, X, y, cv=5, scoring='accuracy')
    sk_cv_time = time.time() - t0

    print(f"Accuracy по фолдам: {sk_scores}")
    print(f"Среднее accuracy:   {np.mean(sk_scores):.4f} (+/- {np.std(sk_scores):.4f})")
    print(f"Время:              {sk_cv_time:.4f} сек")

    # обучаю обе модели на всех данных и замеряю время
    print("\n-- Замер времени обучения на полных данных --")

    t0 = time.time()
    my_model = GradientBoostingCustom(**params)
    my_model.fit(X, y)
    my_fit_time = time.time() - t0

    t0 = time.time()
    sk_model.fit(X, y)
    sk_fit_time = time.time() - t0

    my_train_acc = accuracy_score(y, my_model.predict(X))
    sk_train_acc = accuracy_score(y, sk_model.predict(X))

    print(f"Моя реализация:  accuracy={my_train_acc:.4f}, время={my_fit_time:.4f} сек")
    print(f"sklearn:         accuracy={sk_train_acc:.4f}, время={sk_fit_time:.4f} сек")

    # разные learning_rate
    print("\n-- Влияние learning_rate --")
    print(f"{'learning_rate':<15} {'CV accuracy':<15} {'std':<10}")
    print("-" * 40)
    lr_grid = [0.01, 0.05, 0.1, 0.2, 0.5]
    lr_means, lr_stds = [], []
    for lr in lr_grid:
        p = params.copy()
        p['learning_rate'] = lr
        scores = cross_validate_manual(GradientBoostingCustom, X, y, p, n_folds=5)
        lr_means.append(np.mean(scores))
        lr_stds.append(np.std(scores))
        print(f"{lr:<15.2f} {np.mean(scores):<15.4f} {np.std(scores):<10.4f}")

    # разное число деревьев
    print("\n-- Влияние n_estimators --")
    print(f"{'n_estimators':<15} {'CV accuracy':<15} {'std':<10}")
    print("-" * 40)
    n_grid = [10, 50, 100, 200, 300]
    n_means, n_stds = [], []
    for n_est in n_grid:
        p = params.copy()
        p['n_estimators'] = n_est
        scores = cross_validate_manual(GradientBoostingCustom, X, y, p, n_folds=5)
        n_means.append(np.mean(scores))
        n_stds.append(np.std(scores))
        print(f"{n_est:<15} {np.mean(scores):<15.4f} {np.std(scores):<10.4f}")

    # итоговая таблица
    print("\n" + "=" * 60)
    print("  Сравнение результатов")
    print("=" * 60)
    print(f"{'Метрика':<30} {'Моя реализация':<20} {'sklearn':<15}")
    print("-" * 65)
    print(f"{'CV Accuracy (среднее)':<30} {np.mean(my_scores):<20.4f} {np.mean(sk_scores):<15.4f}")
    print(f"{'CV Accuracy (std)':<30} {np.std(my_scores):<20.4f} {np.std(sk_scores):<15.4f}")
    print(f"{'Время CV (сек)':<30} {my_cv_time:<20.4f} {sk_cv_time:<15.4f}")
    print(f"{'Время обучения (сек)':<30} {my_fit_time:<20.4f} {sk_fit_time:<15.4f}")
    print(f"{'Train accuracy':<30} {my_train_acc:<20.4f} {sk_train_acc:<15.4f}")

    # ============================================================
    #  Графики
    # ============================================================

    # 1. кривая log-loss по итерациям обучения
    loss_model = GradientBoostingCustom(**params)
    loss_model.fit(X, y, track_loss=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_model.loss_history) + 1), loss_model.loss_history,
            color='#d62728', linewidth=2)
    ax.set_xlabel('Итерация (число деревьев)')
    ax.set_ylabel('Log-loss')
    ax.set_title('Сходимость градиентного бустинга')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'loss_curve.png'), dpi=100)
    plt.close()

    # 2. влияние learning_rate
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(lr_grid, lr_means, yerr=lr_stds, marker='o',
                color='#1f77b4', linewidth=2, capsize=4)
    ax.set_xscale('log')
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('CV accuracy')
    ax.set_title('Влияние learning_rate на качество (5-fold CV)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'lr_effect.png'), dpi=100)
    plt.close()

    # 3. влияние n_estimators
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(n_grid, n_means, yerr=n_stds, marker='o',
                color='#2ca02c', linewidth=2, capsize=4)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('CV accuracy')
    ax.set_title('Влияние числа деревьев на качество (5-fold CV)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'n_estimators_effect.png'), dpi=100)
    plt.close()

    # 4. сравнение моя vs sklearn (CV accuracy по фолдам)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w / 2, my_scores, w, label='Моя реализация', color='#1f77b4')
    ax.bar(x + w / 2, sk_scores, w, label='sklearn', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel('Accuracy')
    ax.set_title('CV accuracy по фолдам (5-fold)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'cv_folds_comparison.png'), dpi=100)
    plt.close()

    print(f"\nГрафики сохранены в {IMAGES_DIR}")
