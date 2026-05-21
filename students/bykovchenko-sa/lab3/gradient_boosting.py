
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, f1_score)
from sklearn.ensemble import (GradientBoostingRegressor,GradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MyGradientBoosting:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        loss='squared_error',
        min_samples_split=2,
        min_samples_leaf=5,
        early_stopping_rounds=None,
        validation_fraction=0.2,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.loss_name = loss
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        self.trees = []
        self.alphas = []
        self.initial_prediction = None
        self.best_iteration_ = None

    def _init_prediction(self, y):
        """
        Для регрессии a_0 = mean(y)
        Для классификации a_0 = log(p/(1-p))
        """
        if self.loss_name == 'squared_error':
            return np.mean(y)
        elif self.loss_name == 'log_loss':
            p = np.mean(y == 1)
            p = np.clip(p, 0.01, 0.99)
            return np.log(p / (1.0 - p))
        return 0.0

    def _negative_gradient(self, y, y_pred):
        if self.loss_name == 'squared_error':
            return y - y_pred
        elif self.loss_name == 'log_loss':
            return y / (1.0 + np.exp(y * y_pred))
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

    def _line_search_alpha(self, X, y, y_pred, tree):
        residuals = self._negative_gradient(y, y_pred)
        tree_pred = tree.predict(X)

        if self.loss_name == 'squared_error':
            numerator = np.sum(residuals * tree_pred)
            denominator = np.sum(tree_pred ** 2)
            if abs(denominator) > 1e-10:
                return numerator / denominator
            return 1.0

        elif self.loss_name == 'log_loss':
            sigmoid = 1.0 / (1.0 + np.exp(np.clip(-y * y_pred, -500, 500)))
            hessian = sigmoid * (1.0 - sigmoid)

            numerator = np.sum(residuals * tree_pred)
            denominator = np.sum(hessian * tree_pred ** 2) + 1.0

            if abs(denominator) > 1e-10:
                alpha = numerator / denominator
                return np.clip(alpha, 0.001, 10.0)
            return 0.1

    def fit(self, X, y):
        n_samples = X.shape[0]

        if self.loss_name == 'log_loss':
            y_internal = np.where(y == 0, -1, 1).astype(float)
        else:
            y_internal = y.copy().astype(float)

        self.initial_prediction = self._init_prediction(y_internal)
        y_pred = np.full(n_samples, self.initial_prediction, dtype=float)

        # Early stopping: откладываем валидацию
        X_train, y_train_internal = X, y_internal
        X_val, y_val_internal = None, None
        if self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
            val_size = max(int(n_samples * self.validation_fraction), 10)
            perm = np.random.permutation(n_samples)
            X_train = X[perm[val_size:]]
            y_train_internal = y_internal[perm[val_size:]]
            X_val = X[perm[:val_size]]
            y_val_internal = y_internal[perm[:val_size]]

            # Пересчитываем y_pred только для train
            y_pred = np.full(X_train.shape[0], self.initial_prediction, dtype=float)

        n_train = X_train.shape[0]
        self.trees = []
        self.alphas = []
        best_val_score = None
        best_round = 0

        for t in range(self.n_estimators):
            # Шаг 1 антиградиент
            neg_grad = self._negative_gradient(y_train_internal, y_pred)

            # Шаг 2 стохастическое сэмплирование
            if self.subsample < 1.0:
                sample_size = int(n_train * self.subsample)
                idx = np.random.choice(n_train, size=sample_size, replace=False)
            else:
                idx = np.arange(n_train)

            # Шаг 3 Обучаем b_t
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)
            tree.fit(X_train[idx], neg_grad[idx])

            # Шаг 4 оптимальный α_t
            alpha = self._line_search_alpha(X_train[idx], y_train_internal[idx], y_pred[idx], tree)

            # Шаг 5 обновление
            effective_alpha = self.learning_rate * alpha
            y_pred += effective_alpha * tree.predict(X_train)

            self.trees.append(tree)
            self.alphas.append(effective_alpha)

            # early stopping check
            if X_val is not None and self.early_stopping_rounds is not None:
                val_pred = self._raw_predict(X_val)
                if self.loss_name == 'log_loss':
                    val_pred = np.clip(val_pred, -500, 500)
                    val_loss = np.mean(np.log(1.0 + np.exp(-y_val_internal * val_pred)))
                else:
                    val_loss = mean_squared_error(y_val_internal, val_pred)

                if best_val_score is None or val_loss < best_val_score:
                    best_val_score = val_loss
                    best_round = t
                elif t - best_round >= self.early_stopping_rounds:
                    # обрезаем лишние деревья
                    self.trees = self.trees[:best_round + 1]
                    self.alphas = self.alphas[:best_round + 1]
                    self.best_iteration_ = best_round + 1
                    break

        return self

    def _raw_predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction, dtype=float)
        for tree, alpha in zip(self.trees, self.alphas):
            y_pred += alpha * tree.predict(X)
        return y_pred

    def predict(self, X):
        raw = self._raw_predict(X)

        if self.loss_name == 'log_loss':
            raw = np.clip(raw, -500, 500)
            return np.where(raw >= 0, 1, 0)
        else:
            return raw

    def predict_proba(self, X):
        if self.loss_name != 'log_loss':
            raise ValueError("predict_proba only for loss='log_loss'")
        raw = np.clip(self._raw_predict(X), -500, 500)
        proba_1 = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1 - proba_1, proba_1])


print("-" * 70)
print("Регрессия — California Housing")
print("-" * 70)
print()

housing = fetch_california_housing()
X_reg_full, y_reg_full = housing.data, housing.target
sample_idx = np.random.choice(len(X_reg_full), size=min(5000, len(X_reg_full)), replace=False)
X_reg, y_reg = X_reg_full[sample_idx], y_reg_full[sample_idx]
print(f"Размерность: {X_reg.shape[0]} объектов, {X_reg.shape[1]} признаков")
print(f"Признаки: {', '.join(housing.feature_names)}")
print(f"Целевая переменная: Median house value")
print()

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
print(f"Обучающая выборка: {X_reg_train.shape[0]} объектов")
print(f"Тестовая выборка: {X_reg_test.shape[0]} объектов")
print()

print("-" * 70)
print("Классификация — Breast Cancer Wisconsin")
print("-" * 70)
print()

cancer = load_breast_cancer()
X_clf, y_clf = cancer.data, cancer.target
print(f"Размерность: {X_clf.shape[0]} объектов, {X_clf.shape[1]} признаков")
print(f"Классы: {cancer.target_names} ({np.bincount(y_clf)})")
print()

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f"Обучающая выборка: {X_clf_train.shape[0]} объектов")
print(f"Тестовая выборка: {X_clf_test.shape[0]} объектов")
print()


# Обучение и оценка
print("=" * 70)
print("Регрессия")
print("=" * 70)
print()

N_ESTIMATORS = 100
MAX_DEPTH = 3
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8

print(f"Параметры: T={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
      f"learning_rate={LEARNING_RATE}, subsample={SUBSAMPLE}")
print()

print("Обучение MyGradientBoosting...")
t0 = time.time()
gb_custom_reg = MyGradientBoosting(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                                   subsample=SUBSAMPLE, loss='squared_error', min_samples_leaf=5,
                                   early_stopping_rounds=10, random_state=4)
gb_custom_reg.fit(X_reg_train, y_reg_train)
time_custom_reg = time.time() - t0
print(f"    Время обучения: {time_custom_reg:.3f} с")
if gb_custom_reg.best_iteration_:
    print(f"    Early stopping на итерации: {gb_custom_reg.best_iteration_}")

y_pred_custom_reg = gb_custom_reg.predict(X_reg_test)
mse_custom = mean_squared_error(y_reg_test, y_pred_custom_reg)
r2_custom = r2_score(y_reg_test, y_pred_custom_reg)
print(f"    MSE={mse_custom:.4f}")
print(f"    R²={r2_custom:.4f}")
print()

print("Обучение sklearn GradientBoostingRegressor...")
t0 = time.time()
gb_sklearn_reg = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                                           subsample=SUBSAMPLE, loss='squared_error', random_state=42)
gb_sklearn_reg.fit(X_reg_train, y_reg_train)
time_sklearn_reg = time.time() - t0
print(f"    Время обучения: {time_sklearn_reg:.3f} с")

y_pred_sklearn_reg = gb_sklearn_reg.predict(X_reg_test)
mse_sklearn = mean_squared_error(y_reg_test, y_pred_sklearn_reg)
r2_sklearn = r2_score(y_reg_test, y_pred_sklearn_reg)
print(f"    MSE={mse_sklearn:.4f}")
print(f"    R²={r2_sklearn:.4f}")
print()

print("Кросс-валидация (2-fold)...")
kf = KFold(n_splits=2, shuffle=True, random_state=42)

cv_folds_custom_reg = []
cv_folds_sklearn_reg = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_reg)):
    X_tr, X_val = X_reg[train_idx], X_reg[val_idx]
    y_tr, y_val = y_reg[train_idx], y_reg[val_idx]

    gb_cv_c = MyGradientBoosting(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                                 subsample=SUBSAMPLE, loss='squared_error', min_samples_leaf=5,
                                 early_stopping_rounds=10, random_state=42)
    gb_cv_c.fit(X_tr, y_tr)
    cv_folds_custom_reg.append(mean_squared_error(y_val, gb_cv_c.predict(X_val)))

    gb_cv_s = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                                        subsample=SUBSAMPLE, loss='squared_error', random_state=42)
    gb_cv_s.fit(X_tr, y_tr)
    cv_folds_sklearn_reg.append(mean_squared_error(y_val, gb_cv_s.predict(X_val)))

cv_scores_custom_arr = np.array(cv_folds_custom_reg)
cv_scores_sklearn_arr = np.array(cv_folds_sklearn_reg)
cv_mse_custom = cv_scores_custom_arr.mean()
cv_mse_sklearn = cv_scores_sklearn_arr.mean()

print(f"    My GB — CV MSE: {cv_mse_custom:.4f} ± {cv_scores_custom_arr.std():.4f}")
print(f"    Sklearn GB — CV MSE: {cv_mse_sklearn:.4f} ± {cv_scores_sklearn_arr.std():.4f}")
print()

print("Итоги (регрессия):")
print()
print(f"  {'Метрика':<20} {'My GB':<18} {'Sklearn GB':<18} {'Разница':<12}")
print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*12}")
print(f"  {'MSE (тест)':<20} {mse_custom:<18.4f} {mse_sklearn:<18.4f} {mse_custom-mse_sklearn:<+12.4f}")
print(f"  {'R² (тест)':<20} {r2_custom:<18.4f} {r2_sklearn:<18.4f} {r2_custom-r2_sklearn:<+12.4f}")
print(f"  {'CV MSE (2-fold)':<20} {cv_mse_custom:<18.4f} {cv_mse_sklearn:<18.4f} {cv_mse_custom-cv_mse_sklearn:<+12.4f}")
print(f"  {'Время обучения (с)':<20} {time_custom_reg:<18.3f} {time_sklearn_reg:<18.3f} {time_custom_reg-time_sklearn_reg:<+12.3f}")
print()


print("=" * 70)
print("Классификация")
print("=" * 70)
print()

print("Обучение MyGradientBoosting (log_loss)...")
t0 = time.time()
gb_custom_clf = MyGradientBoosting(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                                   subsample=SUBSAMPLE, loss='log_loss', min_samples_leaf=5,
                                   early_stopping_rounds=None, random_state=42)
gb_custom_clf.fit(X_clf_train, y_clf_train)
time_custom_clf = time.time() - t0
print(f"    Время обучения: {time_custom_clf:.3f} с")

y_pred_custom_clf = gb_custom_clf.predict(X_clf_test)
acc_custom = accuracy_score(y_clf_test, y_pred_custom_clf)
f1_custom = f1_score(y_clf_test, y_pred_custom_clf, average='weighted')
print(f"    Accuracy={acc_custom:.4f}")
print(f"    F1-score={f1_custom:.4f}")
print()

print("Обучение sklearn GradientBoostingClassifier...")
t0 = time.time()
gb_sklearn_clf = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                                            subsample=SUBSAMPLE, loss='log_loss', random_state=42)
gb_sklearn_clf.fit(X_clf_train, y_clf_train)
time_sklearn_clf = time.time() - t0
print(f"    Время обучения: {time_sklearn_clf:.3f} с")

y_pred_sklearn_clf = gb_sklearn_clf.predict(X_clf_test)
acc_sklearn = accuracy_score(y_clf_test, y_pred_sklearn_clf)
f1_sklearn = f1_score(y_clf_test, y_pred_sklearn_clf, average='weighted')
print(f"    Accuracy={acc_sklearn:.4f}")
print(f"    F1-score={f1_sklearn:.4f}")
print()

print("Кросс-валидация (2-fold)...")
cv_folds_custom_clf = []
cv_folds_sklearn_clf = []
kf_clf = KFold(n_splits=2, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf_clf.split(X_clf)):
    X_tr, X_val = X_clf[train_idx], X_clf[val_idx]
    y_tr, y_val = y_clf[train_idx], y_clf[val_idx]

    gb_cv_c = MyGradientBoosting(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                                 subsample=SUBSAMPLE, loss='log_loss', min_samples_leaf=5,
                                 early_stopping_rounds=None, random_state=42)
    gb_cv_c.fit(X_tr, y_tr)
    cv_folds_custom_clf.append(accuracy_score(y_val, gb_cv_c.predict(X_val)))

    gb_cv_s = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                                         subsample=SUBSAMPLE, loss='log_loss', random_state=42)
    gb_cv_s.fit(X_tr, y_tr)
    cv_folds_sklearn_clf.append(accuracy_score(y_val, gb_cv_s.predict(X_val)))

cv_scores_custom_clf_arr = np.array(cv_folds_custom_clf)
cv_scores_sklearn_clf_arr = np.array(cv_folds_sklearn_clf)

print(f"    My GB — CV Accuracy: {cv_scores_custom_clf_arr.mean():.4f} ± {cv_scores_custom_clf_arr.std():.4f}")
print(f"    Sklearn GB — CV Accuracy: {cv_scores_sklearn_clf_arr.mean():.4f} ± {cv_scores_sklearn_clf_arr.std():.4f}")
print()

print("Итоги (классификация):")
print()
print(f"  {'Метрика':<24} {'My GB':<18} {'Sklearn GB':<18} {'Разница':<12}")
print(f"  {'-'*24} {'-'*18} {'-'*18} {'-'*12}")
print(f"  {'Accuracy (тест)':<24} {acc_custom:<18.4f} {acc_sklearn:<18.4f} {acc_custom-acc_sklearn:<+12.4f}")
print(f"  {'F1 weighted (тест)':<24} {f1_custom:<18.4f} {f1_sklearn:<18.4f} {f1_custom-f1_sklearn:<+12.4f}")
print(f"  {'CV Accuracy (2-fold)':<24} {cv_scores_custom_clf_arr.mean():<18.4f} {cv_scores_sklearn_clf_arr.mean():<18.4f} {cv_scores_custom_clf_arr.mean()-cv_scores_sklearn_clf_arr.mean():<+12.4f}")
print(f"  {'Время обучения (с)':<24} {time_custom_clf:<18.3f} {time_sklearn_clf:<18.3f} {time_custom_clf-time_sklearn_clf:<+12.3f}")
print()

print("=" * 70)
print("Эксперименты с гиперпараметрами")
print("=" * 70)
print()

print("Эксперимент 1. Зависимость качества от числа деревьев T")
print("Обобщающая способность бустинга не ухудшается с ростом T?")
print()
print(f"  {'T':<8} {'R² (My GB)':<18} {'R² (sklearn)':<18} {'MSE (My GB)':<18}")
print(f"  {'-'*8} {'-'*18} {'-'*18} {'-'*18}")

for T in [10, 50, 100, 200]:
    gb_t = MyGradientBoosting(n_estimators=T, learning_rate=0.1, max_depth=3, subsample=0.8, loss='squared_error',
                              min_samples_leaf=5, early_stopping_rounds=None, random_state=42)
    gb_t.fit(X_reg_train, y_reg_train)
    r2_t = r2_score(y_reg_test, gb_t.predict(X_reg_test))
    mse_t = mean_squared_error(y_reg_test, gb_t.predict(X_reg_test))

    gb_s = GradientBoostingRegressor(n_estimators=T, learning_rate=0.1, max_depth=3, subsample=0.8,
                                     loss='squared_error', random_state=42)
    gb_s.fit(X_reg_train, y_reg_train)
    r2_s = r2_score(y_reg_test, gb_s.predict(X_reg_test))

    print(f"    {T:<8} {r2_t:<18.4f} {r2_s:<18.4f} {mse_t:<18.4f}")
print()


print("Эксперимент 2. Влияние subsample на качество")
print("Оптимально сэмплировать около 60–80% выборки?")
print()
print(f"  {'Subsample':<12} {'R² (My GB)':<18} {'R² (sklearn)':<18}")
print(f"  {'-'*12} {'-'*18} {'-'*18}")

for ss in [0.5, 0.7, 0.8, 1.0]:
    gb_c = MyGradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=ss, loss='squared_error',
                              min_samples_leaf=5, early_stopping_rounds=None, random_state=42)
    gb_c.fit(X_reg_train, y_reg_train)
    r2_c = r2_score(y_reg_test, gb_c.predict(X_reg_test))

    gb_s = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=ss,
                                     loss='squared_error', random_state=42)
    gb_s.fit(X_reg_train, y_reg_train)
    r2_s = r2_score(y_reg_test, gb_s.predict(X_reg_test))
    print(f"  {ss:<12} {r2_c:<18.4f} {r2_s:<18.4f}")
print()


print("Эксперимент 3. Влияние learning_rate")
print()
print(f"  {'LR':<8} {'R² (My GB)':<18} {'R² (sklearn)':<18}")
print(f"  {'-'*8} {'-'*18} {'-'*18}")

for lr in [0.05, 0.1, 0.3, 0.5]:
    n_est = 100
    gb_c = MyGradientBoosting(n_estimators=n_est, learning_rate=lr, max_depth=3, subsample=0.8,
                              loss='squared_error', min_samples_leaf=5, early_stopping_rounds=None, random_state=42)
    gb_c.fit(X_reg_train, y_reg_train)
    r2_c = r2_score(y_reg_test, gb_c.predict(X_reg_test))

    gb_s = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=3, subsample=0.8,
                                     loss='squared_error', random_state=42)
    gb_s.fit(X_reg_train, y_reg_train)
    r2_s = r2_score(y_reg_test, gb_s.predict(X_reg_test))
    print(f"  {lr:<8} {r2_c:<18.4f} {r2_s:<18.4f}")
print()


SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
print("=" * 70)
print("Строим графики")
print("=" * 70)
print()

print("Кривая обучения (MSE / Accuracy vs итерации)...")

# Регрессия: MSE на тесте по итерациям
my_mse_history = []
y_partial = np.full(X_reg_test.shape[0], gb_custom_reg.initial_prediction)
for tree, alpha in zip(gb_custom_reg.trees, gb_custom_reg.alphas):
    y_partial = y_partial + alpha * tree.predict(X_reg_test)
    my_mse_history.append(mean_squared_error(y_reg_test, y_partial))

sk_mse_history = []
for y_pred_staged in gb_sklearn_reg.staged_predict(X_reg_test):
    sk_mse_history.append(mean_squared_error(y_reg_test, y_pred_staged))

# Классификация: Accuracy на тесте по итерациям
my_acc_history = []
y_partial_clf = np.full(X_clf_test.shape[0], gb_custom_clf.initial_prediction)
for tree, alpha in zip(gb_custom_clf.trees, gb_custom_clf.alphas):
    y_partial_clf = y_partial_clf + alpha * tree.predict(X_clf_test)
    preds = np.where(y_partial_clf >= 0, 1, 0)
    my_acc_history.append(accuracy_score(y_clf_test, preds))

sk_acc_history = []
for y_pred_staged in gb_sklearn_clf.staged_predict(X_clf_test):
    sk_acc_history.append(accuracy_score(y_clf_test, y_pred_staged))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Левый subplot: регрессия
ax1 = axes[0]
ax1.plot(range(1, len(my_mse_history) + 1), my_mse_history,
         'b-o', markersize=2, linewidth=1.5, label='MyGradientBoosting')
ax1.plot(range(1, len(sk_mse_history) + 1), sk_mse_history,
         'r--s', markersize=2, linewidth=1.5, label='sklearn GB', alpha=0.8)
ax1.set_xlabel('Number of trees T', fontsize=11)
ax1.set_ylabel('MSE (test)', fontsize=11)
ax1.set_title('Regression: California Housing', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, max(len(my_mse_history), len(sk_mse_history)))

# Правый subplot: классификация
ax2 = axes[1]
ax2.plot(range(1, len(my_acc_history) + 1), my_acc_history,
         'b-o', markersize=2, linewidth=1.5, label='MyGradientBoosting')
ax2.plot(range(1, len(sk_acc_history) + 1), sk_acc_history,
         'r--s', markersize=2, linewidth=1.5, label='sklearn GB', alpha=0.8)
ax2.set_xlabel('Number of trees T', fontsize=11)
ax2.set_ylabel('Accuracy (test)', fontsize=11)
ax2.set_title('Classification: Breast Cancer', fontsize=12)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, max(len(my_acc_history), len(sk_acc_history)))

plt.tight_layout()
path1 = os.path.join(SAVE_DIR, 'plot1_learning_curve.png')
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {path1}")
print()


print("Scatter plot: y_pred vs y_true (регрессия)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_pred, title, color in [
    (axes[0], y_pred_custom_reg, 'MyGradientBoosting', 'royalblue'),
    (axes[1], y_pred_sklearn_reg, 'sklearn GB', 'crimson')
]:
    ax.scatter(y_reg_test, y_pred, alpha=0.3, s=10, c=color, edgecolors='none')
    lims = [min(y_reg_test.min(), y_pred.min()) - 0.2, max(y_reg_test.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='y_pred = y_true')
    ax.set_xlabel('True value y', fontsize=11)
    ax.set_ylabel('Prediction y_pred', fontsize=11)
    r2 = r2_score(y_reg_test, y_pred)
    mse = mean_squared_error(y_reg_test, y_pred)
    ax.set_title(f'{title}\nR2 = {r2:.4f}, MSE = {mse:.4f}', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
path2 = os.path.join(SAVE_DIR, 'plot2_scatter_regression.png')
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {path2}")
print()


print("Hyperparameter bar charts...")
# Число деревьев T
T_values = [10, 50, 100, 200, 500]
r2_my_T, r2_sk_T = [], []
for T in T_values:
    g1 = MyGradientBoosting(n_estimators=T, learning_rate=0.1, max_depth=3, subsample=0.8, loss='squared_error',
                            min_samples_leaf=5, early_stopping_rounds=10, random_state=42)
    g1.fit(X_reg_train, y_reg_train)
    r2_my_T.append(r2_score(y_reg_test, g1.predict(X_reg_test)))

    g2 = GradientBoostingRegressor(n_estimators=T, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
    g2.fit(X_reg_train, y_reg_train)
    r2_sk_T.append(r2_score(y_reg_test, g2.predict(X_reg_test)))

# Learning rate
lr_values = [0.01, 0.05, 0.1, 0.3, 0.5]
r2_my_lr, r2_sk_lr = [], []
for lr in lr_values:
    ne = max(200, int(100 / lr))
    g1 = MyGradientBoosting(n_estimators=ne, learning_rate=lr, max_depth=3, subsample=0.8, loss='squared_error',
                            min_samples_leaf=5, early_stopping_rounds=15, random_state=42)
    g1.fit(X_reg_train, y_reg_train)
    r2_my_lr.append(r2_score(y_reg_test, g1.predict(X_reg_test)))

    g2 = GradientBoostingRegressor(n_estimators=ne, learning_rate=lr, max_depth=3, subsample=0.8, random_state=42)
    g2.fit(X_reg_train, y_reg_train)
    r2_sk_lr.append(r2_score(y_reg_test, g2.predict(X_reg_test)))

# Max depth
depth_values = [1, 2, 3, 4, 5, 6]
r2_my_d, r2_sk_d = [], []
for d in depth_values:
    g1 = MyGradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=d,
                            subsample=0.8, loss='squared_error', min_samples_leaf=5,
                            early_stopping_rounds=10, random_state=42)
    g1.fit(X_reg_train, y_reg_train)
    r2_my_d.append(r2_score(y_reg_test, g1.predict(X_reg_test)))

    g2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=d, subsample=0.8, random_state=42)
    g2.fit(X_reg_train, y_reg_train)
    r2_sk_d.append(r2_score(y_reg_test, g2.predict(X_reg_test)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
w = 0.35

# Число деревьев T
x_T = np.arange(len(T_values))
axes[0].bar(x_T - w/2, r2_my_T, w, label='MyGradientBoosting', color='royalblue', alpha=0.8)
axes[0].bar(x_T + w/2, r2_sk_T, w, label='sklearn GB', color='crimson', alpha=0.8)
axes[0].set_xlabel('Number of trees T', fontsize=10)
axes[0].set_ylabel('R2 (test)', fontsize=10)
axes[0].set_title('Effect of number of trees T', fontsize=11)
axes[0].set_xticks(x_T)
axes[0].set_xticklabels([str(t) for t in T_values])
axes[0].legend(loc='best', fontsize=9)
axes[0].set_ylim(0.45, 0.85)
axes[0].grid(axis='y', alpha=0.3)

# Learning rate
x_lr = np.arange(len(lr_values))
axes[1].bar(x_lr - w/2, r2_my_lr, w, label='MyGradientBoosting', color='royalblue', alpha=0.8)
axes[1].bar(x_lr + w/2, r2_sk_lr, w, label='sklearn GB', color='crimson', alpha=0.8)
axes[1].set_xlabel('Learning rate', fontsize=10)
axes[1].set_ylabel('R2 (test)', fontsize=10)
axes[1].set_title('Effect of learning rate', fontsize=11)
axes[1].set_xticks(x_lr)
axes[1].set_xticklabels([str(lr) for lr in lr_values])
axes[1].legend(loc='best', fontsize=9)
axes[1].set_ylim(0.45, 0.85)
axes[1].grid(axis='y', alpha=0.3)

# Max depth
x_d = np.arange(len(depth_values))
axes[2].bar(x_d - w/2, r2_my_d, w, label='MyGradientBoosting', color='royalblue', alpha=0.8)
axes[2].bar(x_d + w/2, r2_sk_d, w, label='sklearn GB', color='crimson', alpha=0.8)
axes[2].set_xlabel('Max depth', fontsize=10)
axes[2].set_ylabel('R2 (test)', fontsize=10)
axes[2].set_title('Effect of tree depth', fontsize=11)
axes[2].set_xticks(x_d)
axes[2].set_xticklabels([str(d) for d in depth_values])
axes[2].legend(loc='best', fontsize=9)
axes[2].set_ylim(0.45, 0.85)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
path3 = os.path.join(SAVE_DIR, 'plot3_hyperparams.png')
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {path3}")
print()


print("Residual evolution across iterations...")
iterations_to_show = [0, 10, 50, 100]
iterations_to_show = [t for t in iterations_to_show if t <= len(gb_custom_reg.trees)]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes_flat = axes.flatten()

for idx, n_trees in enumerate(iterations_to_show):
    ax = axes_flat[idx]
    # Предсказание после n_trees деревьев
    y_partial = np.full(X_reg_test.shape[0], gb_custom_reg.initial_prediction)
    for i in range(n_trees):
        y_partial = y_partial + gb_custom_reg.alphas[i] * gb_custom_reg.trees[i].predict(X_reg_test)
    residuals = y_reg_test - y_partial

    ax.hist(residuals, bins=40, color='royalblue', alpha=0.7, edgecolor='white', density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='residual = 0')
    mse = mean_squared_error(y_reg_test, y_partial)
    ax.set_xlabel('Residual (y - a_T(x))', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'After {n_trees} trees\nMSE = {mse:.4f}, std = {np.std(residuals):.4f}', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Residual Evolution', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
path4 = os.path.join(SAVE_DIR, 'plot4_residuals.png')
plt.savefig(path4, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {path4}")
print()


print("Feature importance...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# My GB средняя важность по всем деревьям
my_importances = np.zeros(X_reg_train.shape[1])
for tree in gb_custom_reg.trees:
    my_importances += tree.feature_importances_
my_importances /= len(gb_custom_reg.trees)

# Sklearn
sk_importances = gb_sklearn_reg.feature_importances_

# Сортируем по важности sklearn
sorted_idx = np.argsort(sk_importances)[::-1]
feature_names = housing.feature_names

# Левый: My GB
sorted_my = my_importances[sorted_idx]
axes[0].barh(range(len(sorted_idx)), sorted_my, color='royalblue', alpha=0.8)
axes[0].set_yticks(range(len(sorted_idx)))
axes[0].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
axes[0].set_xlabel('Importance', fontsize=11)
axes[0].set_title('MyGradientBoosting\n(averaged over T trees)', fontsize=11)
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Правый: Sklearn
sorted_sk = sk_importances[sorted_idx]
axes[1].barh(range(len(sorted_idx)), sorted_sk, color='crimson', alpha=0.8)
axes[1].set_yticks(range(len(sorted_idx)))
axes[1].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
axes[1].set_xlabel('Importance', fontsize=11)
axes[1].set_title('sklearn GradientBoostingRegressor', fontsize=11)
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Feature Importance',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
path5 = os.path.join(SAVE_DIR, 'plot5_feature_importance.png')
plt.savefig(path5, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {path5}")
print()
