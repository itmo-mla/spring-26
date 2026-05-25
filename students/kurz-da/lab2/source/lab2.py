import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ============================================================
#  Random Forest (собственная реализация)
# ============================================================

class RandomForestCustom:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt',
                 min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        self.oob_indices = []

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_features

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        m = self._get_max_features(n_features)

        self.trees = []
        self.feature_indices = []
        self.oob_indices = []
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        # для OOB
        self.oob_predictions = np.zeros((n_samples, self.n_classes_))
        self.oob_counts = np.zeros(n_samples)

        for i in range(self.n_estimators):
            # bootstrap-выборка
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(boot_idx))

            # случайное подпространство признаков
            feat_idx = np.sort(np.random.choice(n_features, size=m, replace=False))

            X_boot = X[boot_idx][:, feat_idx]
            y_boot = y[boot_idx]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i
            )
            tree.fit(X_boot, y_boot)

            self.trees.append(tree)
            self.feature_indices.append(feat_idx)
            self.oob_indices.append(oob_idx)

            # считаю OOB-предсказания
            if len(oob_idx) > 0:
                X_oob = X[oob_idx][:, feat_idx]
                proba = tree.predict_proba(X_oob)
                # выравниваю размерности если дерево видело не все классы
                full_proba = np.zeros((len(oob_idx), self.n_classes_))
                for ci, c in enumerate(tree.classes_):
                    class_pos = np.where(self.classes_ == c)[0][0]
                    full_proba[:, class_pos] = proba[:, ci]
                self.oob_predictions[oob_idx] += full_proba
                self.oob_counts[oob_idx] += 1

        return self

    def predict(self, X):
        # голосование большинством
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes_))

        for tree, feat_idx in zip(self.trees, self.feature_indices):
            proba = tree.predict_proba(X[:, feat_idx])
            for ci, c in enumerate(tree.classes_):
                class_pos = np.where(self.classes_ == c)[0][0]
                votes[:, class_pos] += proba[:, ci]

        return self.classes_[np.argmax(votes, axis=1)]

    def get_oob_score(self):
        # OOB accuracy
        mask = self.oob_counts > 0
        if np.sum(mask) == 0:
            return 0.0
        oob_pred = self.classes_[np.argmax(self.oob_predictions[mask], axis=1)]
        return np.mean(oob_pred == y_train_global[mask])

    def get_feature_importance(self, X, y):
        # OOB permutation importance
        n_features = X.shape[1]
        importances = np.zeros(n_features)

        base_oob = self.get_oob_score()

        for j in range(n_features):
            scores = []
            for t_idx, (tree, feat_idx, oob_idx) in enumerate(
                    zip(self.trees, self.feature_indices, self.oob_indices)):
                if len(oob_idx) == 0:
                    continue
                if j not in feat_idx:
                    continue

                X_oob = X[oob_idx].copy()
                y_oob = y[oob_idx]

                # базовая точность этого дерева на OOB
                pred_base = tree.predict(X_oob[:, feat_idx])
                acc_base = np.mean(pred_base == y_oob)

                # перемешиваю j-й признак
                local_j = np.where(feat_idx == j)[0][0]
                X_oob_perm = X_oob.copy()
                np.random.shuffle(X_oob_perm[:, j])
                pred_perm = tree.predict(X_oob_perm[:, feat_idx])
                acc_perm = np.mean(pred_perm == y_oob)

                scores.append(acc_base - acc_perm)

            if len(scores) > 0:
                importances[j] = np.mean(scores)

        return importances


# ============================================================
#  Основной скрипт
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  Лабораторная работа №2. Ансамбли моделей")
    print("  Random Forest (собственная реализация)")
    print("=" * 60)

    # загружаю данные
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"\nДатасет: Breast Cancer Wisconsin")
    print(f"Размер: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print(f"Классы: {np.unique(y)}, баланс: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # глобальная ссылка для OOB
    y_train_global = y_train

    print(f"Тренировочная выборка: {len(y_train)}")
    print(f"Тестовая выборка:      {len(y_test)}")

    # подбор гиперпараметров через grid search по OOB
    print("\n-- Grid Search по OOB --")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'max_features': ['sqrt', 'log2'],
    }

    best_oob = -1
    best_params = {}
    grid_records = []

    for n_est in param_grid['n_estimators']:
        for md in param_grid['max_depth']:
            for mf in param_grid['max_features']:
                rf = RandomForestCustom(
                    n_estimators=n_est, max_depth=md,
                    max_features=mf, random_state=42
                )
                rf.fit(X_train, y_train)
                oob = rf.get_oob_score()
                grid_records.append({
                    'n_estimators': n_est, 'max_depth': md,
                    'max_features': mf, 'oob': oob,
                })
                if oob > best_oob:
                    best_oob = oob
                    best_params = {'n_estimators': n_est, 'max_depth': md, 'max_features': mf}

    print(f"Лучшие параметры: {best_params}")
    print(f"Лучший OOB-score: {best_oob:.4f}")

    # обучаю лучшую модель
    print("\n-- Обучение лучшей модели --")
    t0 = time.time()
    best_rf = RandomForestCustom(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)
    my_time = time.time() - t0

    y_pred = best_rf.predict(X_test)
    my_acc = accuracy_score(y_test, y_pred)
    my_f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {my_acc:.4f}")
    print(f"F1-score: {my_f1:.4f}")
    print(f"Время обучения: {my_time:.4f} сек")

    # важность признаков
    print("\n-- Важность признаков (OOB permutation) --")
    importances = best_rf.get_feature_importance(X_train, y_train)
    sorted_idx = np.argsort(importances)[::-1]

    print(f"{'Признак':<35} {'Важность':<10}")
    print("-" * 45)
    for i in sorted_idx[:10]:
        print(f"{feature_names[i]:<35} {importances[i]:.4f}")

    # sklearn для сравнения
    print("\n-- Эталон: sklearn RandomForestClassifier --")
    sk_params = {
        'n_estimators': best_params['n_estimators'],
        'max_depth': best_params['max_depth'],
        'max_features': best_params['max_features'],
        'random_state': 42
    }
    sk_rf = RandomForestClassifier(**sk_params)

    t0 = time.time()
    sk_rf.fit(X_train, y_train)
    sk_time = time.time() - t0

    y_pred_sk = sk_rf.predict(X_test)
    sk_acc = accuracy_score(y_test, y_pred_sk)
    sk_f1 = f1_score(y_test, y_pred_sk)

    print(f"Accuracy: {sk_acc:.4f}")
    print(f"F1-score: {sk_f1:.4f}")
    print(f"Время обучения: {sk_time:.4f} сек")

    # итоговая таблица
    print("\n" + "=" * 60)
    print("  Сравнение результатов")
    print("=" * 60)
    print(f"{'Метрика':<25} {'Моя реализация':<20} {'sklearn':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {my_acc:<20.4f} {sk_acc:<15.4f}")
    print(f"{'F1-score':<25} {my_f1:<20.4f} {sk_f1:<15.4f}")
    print(f"{'OOB-score':<25} {best_oob:<20.4f} {sk_rf.oob_score_ if hasattr(sk_rf, 'oob_score_') else 'N/A':<15}")
    print(f"{'Время обучения (сек)':<25} {my_time:<20.4f} {sk_time:<15.4f}")

    # ============================================================
    #  Графики
    # ============================================================

    # 1. OOB-score vs n_estimators (для max_features=sqrt, max_depth=10)
    n_grid = [10, 25, 50, 75, 100, 150, 200, 300]
    oob_curve = []
    for n_est in n_grid:
        rf_curve = RandomForestCustom(
            n_estimators=n_est, max_depth=10,
            max_features='sqrt', random_state=42
        )
        rf_curve.fit(X_train, y_train)
        oob_curve.append(rf_curve.get_oob_score())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_grid, oob_curve, marker='o', color='#1f77b4', linewidth=2)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('OOB-score')
    ax.set_title('Зависимость OOB-score от числа деревьев')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'oob_vs_n_estimators.png'), dpi=100)
    plt.close()

    # 2. feature importances (top-15)
    top_n = 15
    top_idx = sorted_idx[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(top_n), importances[top_idx], color='#2ca02c')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.set_xlabel('OOB permutation importance')
    ax.set_title('Топ-15 важных признаков (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), dpi=100)
    plt.close()

    # 3. сравнение метрик my vs sklearn
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['Моя реализация', 'sklearn']
    accs = [my_acc, sk_acc]
    f1s = [my_f1, sk_f1]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, accs, w, label='Accuracy', color='#1f77b4')
    ax.bar(x + w / 2, f1s, w, label='F1-score', color='#ff7f0e')
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w / 2, a + 0.005, f'{a:.3f}', ha='center', fontsize=9)
        ax.text(i + w / 2, f + 0.005, f'{f:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Метрика')
    ax.set_title('Сравнение качества классификации (Breast Cancer)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'metrics_comparison.png'), dpi=100)
    plt.close()

    # 4. влияние max_features (sqrt vs log2) при разных глубинах
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = [5, 10, 15, 'None']
    for mf, color in zip(['sqrt', 'log2'], ['#1f77b4', '#ff7f0e']):
        vals = []
        for md in [5, 10, 15, None]:
            rec = [r for r in grid_records
                   if r['n_estimators'] == best_params['n_estimators']
                   and r['max_depth'] == md and r['max_features'] == mf]
            vals.append(rec[0]['oob'] if rec else np.nan)
        ax.plot(depths, vals, marker='o', label=f'max_features={mf}', color=color, linewidth=2)
    ax.set_xlabel('max_depth')
    ax.set_ylabel('OOB-score')
    ax.set_title(f'Влияние гиперпараметров (n_estimators={best_params["n_estimators"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'oob_vs_depth.png'), dpi=100)
    plt.close()

    print(f"\nГрафики сохранены в {IMAGES_DIR}")
