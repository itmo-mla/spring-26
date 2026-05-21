import numpy as np
import pandas as pd
import time
import os
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve, f1_score)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class MyRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 random_state=None, oob_score=True, bootstrap=True, class_weight=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.oob_score = oob_score
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.estimators_ = []
        self.oob_score_ = None
        self.oob_decision_function_ = None
        self.feature_importances_ = None
        self.n_features_ = None
        self.classes_ = None
        self.oob_samples_ = []

    def _get_max_features(self, n_features):
        """
        Определение количества признаков для каждого сплита.
        max_features = k случайно выбираемых признаков в каждой вершине:
            - k = ⌊n/3⌋ для регрессии
            - k = ⌊√n⌋ для классификации
        """
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features)) # ⌊√n⌋ для классификации
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features

    def _bootstrap_sample(self, n_samples, random_state):
        """Создание бутстреп-выборки"""
        rng = np.random.RandomState(random_state)
        if self.bootstrap:
            # Выборка с возвращением
            indices = rng.randint(0, n_samples, size=n_samples)

            # Определяем out-of-bag сэмплы
            # Tᵢ = {t : xᵢ ∉ Uₜ} — множество деревьев, где xᵢ не попал в выборку
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[indices] = False
            oob_indices = np.where(oob_mask)[0]
        else:
            # Без бутстрепа — все сэмплы используются
            indices = np.arange(n_samples)
            oob_indices = np.array([], dtype=int)
        return indices, oob_indices

    def fit(self, X, y):
        """Обучение ансамбля"""
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        max_features = self._get_max_features(n_features) # k = ⌊√n⌋

        self.estimators_ = []
        self.oob_samples_ = []

        # Инициализация массивов для OOB предсказаний
        if self.oob_score:
            oob_predictions = np.zeros((n_samples, n_classes))
            oob_counts = np.zeros(n_samples)

        rng = np.random.RandomState(self.random_state)

        # Порог отбора: дерево идёт в ансамбль, только если оно лучше случайного угадывания
        # Для классификации случайное угадывание = 1 / n_classes (для бинарной = 0.5)
        select_threshold = 1.0 / n_classes

        # Строим деревья, пока не наберем n_estimators хороших алгоритмов.
        # Используем while, так как плохие деревья будут отбрасываться.
        max_attempts = self.n_estimators * 3  # Защита от бесконечного цикла
        attempts = 0

        while len(self.estimators_) < self.n_estimators and attempts < max_attempts:
            attempts += 1
            tree_seed = rng.randint(0, 2 ** 31 - 1)

            # Бутстреп-выборка
            sample_indices, oob_indices = self._bootstrap_sample(n_samples, tree_seed)

            # Обучение базового алгоритма
            tree = DecisionTreeClassifier(
                max_features=max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
                class_weight=self.class_weight
            )
            tree.fit(X[sample_indices], y[sample_indices])

            # Проверка качества базового алгоритма (Не каждый базовый алгоритм идёт в ансамбль)
            if len(oob_indices) > 0:
                oob_accuracy = tree.score(X[oob_indices], y[oob_indices])
                if oob_accuracy <= select_threshold:
                    # Дерево не лучше случайного — отбрасываем его
                    continue
            else:
                # Если нет OOB-объектов (крайне редкий случай на маленьких выборках), тоже отбрасываем
                continue

            # Дерево прошло отбор — добавляем в ансамбль
            self.estimators_.append(tree)
            self.oob_samples_.append(oob_indices)

            # накапливаем OOB предсказания
            # OOB(xᵢ) = (1/|Tᵢ|) Σₜ∈Tᵢ bₜ(xᵢ)
            if self.oob_score and len(oob_indices) > 0:
                proba = tree.predict_proba(X[oob_indices])

                # согласование порядка классов между деревьями
                if not np.array_equal(tree.classes_, self.classes_):
                    # если классы в другом порядке — переставляем
                    proba_ordered = np.zeros((len(oob_indices), n_classes))
                    for j, cls in enumerate(tree.classes_):
                        idx = np.where(self.classes_ == cls)[0][0]
                        proba_ordered[:, idx] = proba[:, j]
                    proba = proba_ordered
                oob_predictions[oob_indices] += proba
                oob_counts[oob_indices] += 1

        # вычисляем OOB Score
        # OOB(X^ℓ) = Σᵢ₌₁^ℓ L(OOB(xᵢ), yᵢ)
        if self.oob_score:
            valid_mask = oob_counts > 0
            if valid_mask.sum() > 0:
                # Усреднение OOB(xᵢ) = (1/|Tᵢ|) Σₜ∈Tᵢ bₜ(xᵢ)
                oob_predictions[valid_mask]/= oob_counts[valid_mask, np.newaxis]
                self.oob_decision_function_ = oob_predictions

                oob_pred_indices = np.argmax(oob_predictions[valid_mask], axis=1)
                oob_pred_labels = self.classes_[oob_pred_indices]
                oob_true = y[valid_mask]

                # accuracy на OOB сэмплах
                self.oob_score_ = np.mean(oob_pred_labels == oob_true)
            else:
                self.oob_score_ = None

        self._compute_feature_importances()
        return self

    def _compute_feature_importances(self):
        """Вычисление важности признаков (Gini Importance)"""
        importances = np.zeros(self.n_features_)
        for tree in self.estimators_:
            importances+=tree.feature_importances_
        importances/=len(self.estimators_)    # усреднение по T деревьям
        self.feature_importances_ = importances

    def compute_oob_feature_importance(self, X, y, n_repeats=10):
        """
        OOB Permutation Importance.

        Для каждого признака j и каждого дерева t:
        1. Вычислить baseline точность на OOB-объектах
        2. Перемешать значения признака j
        3. Вычислить точность после перемешивания
        4. Важность = разница между baseline и перемешанной точностью
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        importances = np.zeros(n_features)
        rng = np.random.RandomState(self.random_state)

        for j in range(n_features):     # для каждого признака fⱼ
            importance_j = 0
            n_valid_trees = 0
            for i, tree in enumerate(self.estimators_):
                oob_indices = self.oob_samples_[i]
                if len(oob_indices) == 0:
                    continue
                n_valid_trees += 1
                X_oob = X[oob_indices].copy()
                y_oob = y[oob_indices]

                # базовая OOB-точность
                orig_score = tree.score(X_oob, y_oob)

                # перемешиваем признак и вычисляем точность
                permuted_scores = []
                for _ in range(n_repeats):
                    X_permuted = X_oob.copy()
                    perm_indices = rng.permutation(len(oob_indices))
                    X_permuted[:, j] = X_permuted[perm_indices, j]  # перемешивание fⱼ
                    permuted_scores.append(tree.score(X_permuted, y_oob))

                avg_permuted_score = np.mean(permuted_scores)

                # разница OOBⱼ - OOB
                importance_j += (orig_score - avg_permuted_score)

            if n_valid_trees > 0:
                importances[j] = importance_j / n_valid_trees

        # нормализация
        if importances.sum() > 0:
            importances = importances / importances.sum()
        return importances

    def predict(self, X):
        """Предсказание классов, простое голосование"""
        X = np.asarray(X)

        # получаем предсказания всех деревьев b₁(x), ..., bₜ(x)
        predictions = np.array([tree.predict(X) for tree in self.estimators_])

        # Векторизация. Для каждого сэмпла находим наиболее частый класс
        mode_result = stats.mode(predictions, axis=0, keepdims=True)
        majority_vote = mode_result.mode.flatten()

        return majority_vote

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        X = np.asarray(X)
        proba_sum = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X)

            # согласование порядка классов
            if not np.array_equal(tree.classes_, self.classes_):
                proba_ordered = np.zeros((X.shape[0], len(self.classes_)))
                for j, cls in enumerate(tree.classes_):
                    idx = np.where(self.classes_ == cls)[0][0]
                    proba_ordered[:, idx] = tree_proba[:, j]
                tree_proba = proba_ordered
            proba_sum += tree_proba

        # F = (1/T) Σₜ bₜ
        return proba_sum / len(self.estimators_)


def load_data():
    print("Загрузка датасета")
    bank = fetch_openml('bank-marketing', version=2, as_frame=True, parser='auto')

    X = bank.data.copy()
    y = bank.target
    feature_names = list(X.columns)

    # кодирование категориальных признаков
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # кодирование целевой переменной
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")
    print(f"Количество признаков: {X.shape[1]}")

    print(f"\nРаспределение классов:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Класс {cls}: {cnt} ({cnt / len(y_train) * 100:.1f}%)")

    return X_train, X_test, y_train, y_test, feature_names


def grid_search_oob(X_train, y_train):
    """Подбор гиперпараметров по OOB"""
    print("\n" + "=" * 70)
    print("Grid Search по OOB")
    print("=" * 70)

    param_grid = {'n_estimators': [30, 50, 100], 'max_features': ['sqrt', 0.5], 'max_depth': [5, 10, None], 'min_samples_leaf': [1, 5]}
    print("\nПараметры для перебора:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")

    grid_results = []
    param_combinations = list(ParameterGrid(param_grid))
    print(f"\nВсего комбинаций: {len(param_combinations)}")
    print("\nПоиск лучших параметров...")

    for params in param_combinations:
        rf = MyRandomForestClassifier(**params, random_state=42, oob_score=True, class_weight='balanced')
        rf.fit(X_train, y_train)

        grid_results.append({'n_estimators': params['n_estimators'], 'max_features': params['max_features'],
                             'max_depth': params['max_depth'], 'min_samples_leaf': params['min_samples_leaf'],
                             'oob_score': rf.oob_score_})

    # сортируем по OOB Score
    grid_df = pd.DataFrame(grid_results)
    grid_df = grid_df.sort_values('oob_score', ascending=False)

    # берём лучшие параметры из top-1
    best_row = grid_df.iloc[0]
    best_params = {
        'n_estimators': int(best_row['n_estimators']),
        'max_features': best_row['max_features'],
        'max_depth': None if pd.isna(best_row['max_depth']) else int(best_row['max_depth']),
        'min_samples_leaf': int(best_row['min_samples_leaf'])
    }
    best_oob = best_row['oob_score']

    print(f"\nЛучшие параметры: {best_params}")
    print(f"Лучший OOB Score: {best_oob:.4f}")

    return best_params, grid_df


def compare_models(X_train, X_test, y_train, y_test, best_params, feature_names):
    """Сравнение Custom RF с Sklearn RF"""
    print("\n" + "=" * 70)
    print("Custom RF vs Sklearn RF")
    print("=" * 70)

    custom_rf = MyRandomForestClassifier(**best_params, random_state=42, oob_score=True, class_weight='balanced')
    start = time.time()
    custom_rf.fit(X_train, y_train)
    custom_train_time = time.time() - start

    start = time.time()
    y_pred_custom = custom_rf.predict(X_test)
    custom_pred_time = time.time() - start
    custom_acc = accuracy_score(y_test, y_pred_custom)
    y_proba_custom = custom_rf.predict_proba(X_test)[:, 1]

    print("\n" + "-" * 60)
    print("Важность признаков через OOB Permutation Importance:")
    print("-" * 60)
    oob_perm_importance = custom_rf.compute_oob_feature_importance(X_train, y_train, n_repeats=5)
    oob_sorted = sorted(zip(feature_names, oob_perm_importance), key=lambda x: x[1], reverse=True)
    for name, imp in oob_sorted[:5]:
        print(f"  {name}: {imp:.4f}")

    sklearn_rf = SklearnRF(**best_params, random_state=42, oob_score=True, n_jobs=-1, class_weight='balanced')
    start = time.time()
    sklearn_rf.fit(X_train, y_train)
    sklearn_train_time = time.time() - start

    start = time.time()
    y_pred_sklearn = sklearn_rf.predict(X_test)
    sklearn_pred_time = time.time() - start
    sklearn_acc = accuracy_score(y_test, y_pred_sklearn)
    y_proba_sklearn = sklearn_rf.predict_proba(X_test)[:, 1]

    roc_auc_custom = roc_auc_score(y_test, y_proba_custom)
    roc_auc_sklearn = roc_auc_score(y_test, y_proba_sklearn)
    pr_auc_custom = average_precision_score(y_test, y_proba_custom)
    pr_auc_sklearn = average_precision_score(y_test, y_proba_sklearn)

    print("\n" + "-" * 60)
    print(f"{'Метрика':<30} {'Custom RF':<15} {'Sklearn RF':<15}")
    print("-" * 60)
    print(f"{'Время обучения (сек)':<30} {custom_train_time:<15.4f} {sklearn_train_time:<15.4f}")
    print(f"{'Время предсказания (сек)':<30} {custom_pred_time:<15.4f} {sklearn_pred_time:<15.4f}")
    print(f"{'OOB Score':<30} {custom_rf.oob_score_:<15.4f} {sklearn_rf.oob_score_:<15.4f}")
    print(f"{'Точность (Accuracy)':<30} {custom_acc:<15.4f} {sklearn_acc:<15.4f}")
    print(f"{'ROC-AUC':<30} {roc_auc_custom:<15.4f} {roc_auc_sklearn:<15.4f}")
    print(f"{'PR-AUC':<30} {pr_auc_custom:<15.4f} {pr_auc_sklearn:<15.4f}")
    print("-" * 60)

    print("\nВажность признаков (Gini Importance):")
    gini_sorted = sorted(zip(feature_names, custom_rf.feature_importances_), key=lambda x: x[1], reverse=True)
    for name, imp in gini_sorted[:5]:
        print(f"  {name}: {imp:.4f}")

    print("\nДетальные метрики классификации (Custom RF):")
    print(classification_report(y_test, y_pred_custom, target_names=['Class 0', 'Class 1']))

    print("Confusion Matrix (Custom RF):")
    print(confusion_matrix(y_test, y_pred_custom))

    print("\nДетальные метрики классификации (Sklearn RF):")
    print(classification_report(y_test, y_pred_sklearn, target_names=['Class 0', 'Class 1']))

    print("Confusion Matrix (Sklearn RF):")
    print(confusion_matrix(y_test, y_pred_sklearn))

    print("\n" + "=" * 60)
    print("АНАЛИЗ ДИСБАЛАНСА КЛАССОВ")
    print("=" * 60)
    report_custom = classification_report(y_test, y_pred_custom, output_dict=True)
    report_sklearn = classification_report(y_test, y_pred_sklearn, output_dict=True)
    recall_1_custom = report_custom['1']['recall']
    recall_1_sklearn = report_sklearn['1']['recall']
    f1_1_custom = report_custom['1']['f1-score']
    f1_1_sklearn = report_sklearn['1']['f1-score']
    print(f"\n  Recall (Class 1, Custom RF):  {recall_1_custom:.4f}  ← {'низкий!' if recall_1_custom < 0.5 else 'OK'}")
    print(f"  Recall (Class 1, Sklearn RF): {recall_1_sklearn:.4f}  ← {'низкий!' if recall_1_sklearn < 0.5 else 'OK'}")
    print(f"\n  F1-Score (Class 1, Custom RF):  {f1_1_custom:.4f}")
    print(f"  F1-Score (Class 1, Sklearn RF): {f1_1_sklearn:.4f}")

    print("\n" + "=" * 60)
    print("THRESHOLD TUNING (подбор порога по F1)")
    print("=" * 60)
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1_custom = 0
    best_thr_custom = 0.5
    best_f1_sklearn = 0
    best_thr_sklearn = 0.5

    for t in thresholds:
        y_pred_thr_custom = (y_proba_custom >= t).astype(int)
        y_pred_thr_sklearn = (y_proba_sklearn >= t).astype(int)
        f1_c = f1_score(y_test, y_pred_thr_custom)
        f1_s = f1_score(y_test, y_pred_thr_sklearn)
        if f1_c > best_f1_custom:
            best_f1_custom = f1_c
            best_thr_custom = t
        if f1_s > best_f1_sklearn:
            best_f1_sklearn = f1_s
            best_thr_sklearn = t

    print(f"\n  Custom RF:")
    print(f"    Лучший threshold: {best_thr_custom:.3f} (вместо 0.5)")
    print(f"    F1-Score: {best_f1_custom:.4f} (было: {f1_1_custom:.4f})")
    print(f"    Улучшение: +{best_f1_custom - f1_1_custom:.4f}")

    print(f"\n  Sklearn RF:")
    print(f"    Лучший threshold: {best_thr_sklearn:.3f} (вместо 0.5)")
    print(f"    F1-Score: {best_f1_sklearn:.4f} (было: {f1_1_sklearn:.4f})")
    print(f"    Улучшение: +{best_f1_sklearn - f1_1_sklearn:.4f}")

    # предсказания с оптимальным threshold
    y_pred_custom_opt = (y_proba_custom >= best_thr_custom).astype(int)
    print(f"\n  Classification Report с оптимальным threshold (Custom RF):")
    print(classification_report(y_test, y_pred_custom_opt, target_names=['Class 0', 'Class 1']))

    # кривые для графиков
    fpr_custom, tpr_custom, _ = roc_curve(y_test, y_proba_custom)
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, y_proba_sklearn)
    prec_custom, rec_custom, _ = precision_recall_curve(y_test, y_proba_custom)
    prec_sklearn, rec_sklearn, _ = precision_recall_curve(y_test, y_proba_sklearn)

    return {
        'custom': {
            'train_time': custom_train_time,
            'pred_time': custom_pred_time,
            'oob_score': custom_rf.oob_score_,
            'accuracy': custom_acc,
            'roc_auc': roc_auc_custom,
            'pr_auc': pr_auc_custom,
            'feature_importance': dict(zip(feature_names, custom_rf.feature_importances_)),
            'oob_perm_importance': dict(zip(feature_names, oob_perm_importance)),
            'roc_curve': (fpr_custom, tpr_custom),
            'pr_curve': (prec_custom, rec_custom),
            'y_pred': y_pred_custom,
            'y_proba': y_proba_custom,
            'recall_class1': recall_1_custom,
            'f1_class1': f1_1_custom,
            'best_threshold': best_thr_custom,
            'best_f1': best_f1_custom
        },
        'sklearn': {
            'train_time': sklearn_train_time,
            'pred_time': sklearn_pred_time,
            'oob_score': sklearn_rf.oob_score_,
            'accuracy': sklearn_acc,
            'roc_auc': roc_auc_sklearn,
            'pr_auc': pr_auc_sklearn,
            'roc_curve': (fpr_sklearn, tpr_sklearn),
            'pr_curve': (prec_sklearn, rec_sklearn),
            'y_pred': y_pred_sklearn,
            'y_proba': y_proba_sklearn,
            'recall_class1': recall_1_sklearn,
            'f1_class1': f1_1_sklearn,
            'best_threshold': best_thr_sklearn,
            'best_f1': best_f1_sklearn
        }
    }


def create_plots(results, grid_df, feature_names, y_test):
    print("\n" + "=" * 70)

    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    plt.rcParams['figure.figsize'] = (12, 5)

    # График 1. Сравнение времени и качества
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    metrics = ['Training Time (s)', 'Prediction Time (s)']
    custom_vals = [results['custom']['train_time'], results['custom']['pred_time']]
    sklearn_vals = [results['sklearn']['train_time'], results['sklearn']['pred_time']]

    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width / 2, custom_vals, width, label='Custom RF', color='#2196F3')
    ax1.bar(x + width / 2, sklearn_vals, width, label='Sklearn RF', color='#4CAF50')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training and Prediction Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    metrics2 = ['OOB Score', 'Accuracy', 'ROC-AUC', 'PR-AUC']
    custom_vals2 = [results['custom']['oob_score'], results['custom']['accuracy'],
                    results['custom']['roc_auc'], results['custom']['pr_auc']]
    sklearn_vals2 = [results['sklearn']['oob_score'], results['sklearn']['accuracy'],
                     results['sklearn']['roc_auc'], results['sklearn']['pr_auc']]

    x2 = np.arange(len(metrics2))
    ax2.bar(x2 - width / 2, custom_vals2, width, label='Custom RF', color='#2196F3')
    ax2.bar(x2 + width / 2, sklearn_vals2, width, label='Sklearn RF', color='#4CAF50')
    ax2.set_ylabel('Score')
    ax2.set_title('Model Quality Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics2, rotation=15)
    ax2.legend()
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/comparison_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # График 2. Важность признаков (Gini vs OOB Permutation)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gini Importance
    ax1 = axes[0]
    gini_imp = results['custom']['feature_importance']
    gini_sorted = sorted(gini_imp.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in gini_sorted]
    values = [x[1] for x in gini_sorted]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
    ax1.barh(range(len(names)), values, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Importance')
    ax1.set_title('Gini Importance')
    ax1.grid(axis='x', alpha=0.3)

    # OOB Permutation Importance
    ax2 = axes[1]
    oob_imp = results['custom']['oob_perm_importance']
    oob_sorted = sorted(oob_imp.items(), key=lambda x: x[1], reverse=True)
    names2 = [x[0] for x in oob_sorted]
    values2 = [x[1] for x in oob_sorted]
    colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, len(names2)))
    ax2.barh(range(len(names2)), values2, color=colors2)
    ax2.set_yticks(range(len(names2)))
    ax2.set_yticklabels(names2)
    ax2.set_xlabel('Importance')
    ax2.set_title('OOB Permutation Importance')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/feature_importance_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # График 3. ROC и PR кривые
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    fpr_c, tpr_c = results['custom']['roc_curve']
    fpr_s, tpr_s = results['sklearn']['roc_curve']
    ax1.plot(fpr_c, tpr_c, color='#2196F3', lw=2, label=f'Custom RF (AUC = {results["custom"]["roc_auc"]:.4f})')
    ax1.plot(fpr_s, tpr_s, color='#4CAF50', lw=2, label=f'Sklearn RF (AUC = {results["sklearn"]["roc_auc"]:.4f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    prec_c, rec_c = results['custom']['pr_curve']
    prec_s, rec_s = results['sklearn']['pr_curve']
    ax2.plot(rec_c, prec_c, color='#2196F3', lw=2, label=f'Custom RF (AP = {results["custom"]["pr_auc"]:.4f})')
    ax2.plot(rec_s, prec_s, color='#4CAF50', lw=2, label=f'Sklearn RF (AP = {results["sklearn"]["pr_auc"]:.4f})')
    baseline = np.mean(y_test)
    ax2.axhline(y=baseline, color='gray', lw=1, linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # График 4. Grid Search результаты
    fig, ax = plt.subplots(figsize=(10, 6))
    top_results = grid_df.head(20)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_results)))
    ax.barh(range(len(top_results)), top_results['oob_score'], color=colors)
    labels = [f"n={r['n_estimators']}, mf={r['max_features']}, md={r['max_depth']}, msl={r['min_samples_leaf']}"
              for _, r in top_results.iterrows()]
    ax.set_yticks(range(len(top_results)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('OOB Score')
    ax.set_title('Grid Search Results (Top 20)')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/grid_search_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # График 5. Threshold Tuning
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores_custom = []
    f1_scores_sklearn = []

    for t in thresholds:
        y_pred_thr_custom = (results['custom']['y_proba'] >= t).astype(int)
        y_pred_thr_sklearn = (results['sklearn']['y_proba'] >= t).astype(int)
        f1_scores_custom.append(f1_score(y_test, y_pred_thr_custom))
        f1_scores_sklearn.append(f1_score(y_test, y_pred_thr_sklearn))

    ax.plot(thresholds, f1_scores_custom, color='#2196F3', lw=2, label='Custom RF')
    ax.plot(thresholds, f1_scores_sklearn, color='#4CAF50', lw=2, label='Sklearn RF')

    ax.axvline(x=results['custom']['best_threshold'], color='#2196F3', linestyle='--', alpha=0.7,
               label=f'Custom RF best: {results["custom"]["best_threshold"]:.3f}')
    ax.axvline(x=results['sklearn']['best_threshold'], color='#4CAF50', linestyle='--', alpha=0.7,
               label=f'Sklearn RF best: {results["sklearn"]["best_threshold"]:.3f}')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Default: 0.5')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score (Class 1)')
    ax.set_title('Threshold Tuning: F1 Score vs Decision Threshold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0.1, 0.9])

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/threshold_tuning_plot.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_data()
    best_params, grid_df = grid_search_oob(X_train, y_train)
    results = compare_models(X_train, X_test, y_train, y_test, best_params, feature_names)
    create_plots(results, grid_df, feature_names, y_test)

    grid_df.to_csv('grid_search_results.csv', index=False)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gini_importance': [results['custom']['feature_importance'][f] for f in feature_names],
        'oob_permutation_importance': [results['custom']['oob_perm_importance'][f] for f in feature_names]
    }).sort_values('gini_importance', ascending=False)
    importance_df.to_csv('feature_importance.csv', index=False)

    metrics_comparison = pd.DataFrame({
        'metric': ['Training Time (s)', 'Prediction Time (s)', 'OOB Score',
                   'Accuracy', 'ROC-AUC', 'PR-AUC', 'Recall (Class 1)', 'F1 (Class 1)',
                   'Best Threshold', 'F1 (optimized)'],
        'custom_rf': [
            results['custom']['train_time'], results['custom']['pred_time'],
            results['custom']['oob_score'], results['custom']['accuracy'],
            results['custom']['roc_auc'], results['custom']['pr_auc'],
            results['custom']['recall_class1'], results['custom']['f1_class1'],
            results['custom']['best_threshold'], results['custom']['best_f1']
        ],
        'sklearn_rf': [
            results['sklearn']['train_time'], results['sklearn']['pred_time'],
            results['sklearn']['oob_score'], results['sklearn']['accuracy'],
            results['sklearn']['roc_auc'], results['sklearn']['pr_auc'],
            results['sklearn']['recall_class1'], results['sklearn']['f1_class1'],
            results['sklearn']['best_threshold'], results['sklearn']['best_f1']
        ]
    })
    metrics_comparison.to_csv('metrics_comparison.csv', index=False)

    print("\nИТОГИ:")
    print(f"  Лучшие параметры: {best_params}")
    print(f"\n  Custom RF:")
    print(f"    Accuracy:  {results['custom']['accuracy']:.4f}")
    print(f"    ROC-AUC:   {results['custom']['roc_auc']:.4f}")
    print(f"    PR-AUC:    {results['custom']['pr_auc']:.4f}")
    print(f"    Recall (Class 1): {results['custom']['recall_class1']:.4f}")
    print(f"    Best Threshold: {results['custom']['best_threshold']:.3f} → F1: {results['custom']['best_f1']:.4f}")
    print(f"\n  Sklearn RF:")
    print(f"    Accuracy:  {results['sklearn']['accuracy']:.4f}")
    print(f"    ROC-AUC:   {results['sklearn']['roc_auc']:.4f}")
    print(f"    PR-AUC:    {results['sklearn']['pr_auc']:.4f}")
    print(f"    Recall (Class 1): {results['sklearn']['recall_class1']:.4f}")
    print(f"    Best Threshold: {results['sklearn']['best_threshold']:.3f} → F1: {results['sklearn']['best_f1']:.4f}")