import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import time
import warnings
warnings.filterwarnings('ignore')

class RandomForestCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 oob_score=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        self.trees_ = []
        self.oob_indices_ = []
        self.oob_score_ = None
        self.feature_importances_ = None

        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_estimators):
            # Бутстреп выборка
            if self.bootstrap:
                n_samples = X.shape[0]
                indices = rng.choice(n_samples, n_samples, replace=True)
            else:
                indices = np.arange(X.shape[0])

            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(0, 1e6)
            )
            tree.fit(X[indices], y[indices])
            self.trees_.append(tree)
            self.oob_indices_.append(indices)

            # Накопление важности признаков
            if self.feature_importances_ is None:
                self.feature_importances_ = tree.feature_importances_
            else:
                self.feature_importances_ += tree.feature_importances_

        self.feature_importances_ /= self.n_estimators

        if self.oob_score:
            self._compute_oob_error(X, y)
        return self

    def _compute_oob_error(self, X, y):
        n_samples = X.shape[0]
        oob_votes = np.zeros((n_samples, self.n_classes_))
        oob_counts = np.zeros(n_samples, dtype=int)

        for idx, tree in enumerate(self.trees_):
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[self.oob_indices_[idx]] = False
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) == 0:
                continue
            pred = tree.predict_proba(X[oob_idx])
            oob_votes[oob_idx] += pred
            oob_counts[oob_idx] += 1

        oob_pred = np.argmax(oob_votes, axis=1)
        mask = oob_counts > 0
        self.oob_score_ = np.mean(oob_pred[mask] != y[mask])
        self.oob_pred_ = oob_pred
        self.oob_mask_ = oob_counts > 0

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        preds = np.array([tree.predict(X) for tree in self.trees_])
        pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
        return self.classes_[pred]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        probas = np.mean([tree.predict_proba(X) for tree in self.trees_], axis=0)
        return probas

    def feature_importances_permutation(self, X, y, random_state=None):
        if self.oob_score_ is None:
            self._compute_oob_error(X, y)

        rng = np.random.RandomState(random_state)
        base_error = self.oob_score_
        importances = []

        for j in range(self.n_features_):
            error_j = 0.0
            total_weight = 0

            for idx, tree in enumerate(self.trees_):
                oob_mask = np.ones(X.shape[0], dtype=bool)
                oob_mask[self.oob_indices_[idx]] = False
                oob_idx = np.where(oob_mask)[0]
                if len(oob_idx) == 0:
                    continue

                X_oob = X[oob_idx].copy()
                permuted = rng.permutation(X_oob[:, j])
                X_oob[:, j] = permuted

                pred = tree.predict(X_oob)
                err = np.mean(pred != y[oob_idx])
                error_j += err * len(oob_idx)
                total_weight += len(oob_idx)

            if total_weight > 0:
                error_j /= total_weight
                imp = (error_j - base_error) / base_error if base_error > 0 else 0
            else:
                imp = 0
            importances.append(imp)

        return np.array(importances)


data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Классы: {data.target_names}")


print("\nПодбор гиперпараметров по OOB:")

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'max_depth': [5, 10, None]
}

best_params = None
best_oob = 1.0
results = []

for params in ParameterGrid(param_grid):
    start_time = time.time()
    rf = RandomForestCustom(
        n_estimators=params['n_estimators'],
        max_features=params['max_features'],
        max_depth=params['max_depth'],
        oob_score=True,
        random_state=42,
        bootstrap=True
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    oob_err = rf.oob_score_
    results.append({
        **params,
        'oob_error': oob_err,
        'time': elapsed
    })
    if oob_err < best_oob:
        best_oob = oob_err
        best_params = params

results_df = pd.DataFrame(results)
print("\nТоп-10 комбинаций по OOB-ошибке:")
print(results_df.sort_values('oob_error').head(10).to_string(index=False))
print(f"\nЛучшие параметры: {best_params}")
print(f"Лучшая OOB-ошибка: {best_oob:.4f}")

best_n_estimators = best_params['n_estimators']
best_max_features = best_params['max_features']
best_max_depth = best_params['max_depth']


print("\nСравнение с RandomForest из sklearn:")

rf_custom = RandomForestCustom(
    n_estimators=best_n_estimators,
    max_features=best_max_features,
    max_depth=best_max_depth,
    oob_score=True,
    random_state=42,
    bootstrap=True
)
start = time.time()
rf_custom.fit(X_train, y_train)
custom_train_time = time.time() - start
y_pred_custom = rf_custom.predict(X_test)
custom_acc = accuracy_score(y_test, y_pred_custom)

rf_sk = SKRandomForest(
    n_estimators=best_n_estimators,
    max_features=best_max_features,
    max_depth=best_max_depth,
    oob_score=True,
    random_state=42,
    bootstrap=True
)
start = time.time()
rf_sk.fit(X_train, y_train)
sk_train_time = time.time() - start
y_pred_sk = rf_sk.predict(X_test)
sk_acc = accuracy_score(y_test, y_pred_sk)

print(f"Custom RF  : Accuracy={custom_acc:.4f}, Time={custom_train_time:.2f}s")
print(f"sklearn RF  : Accuracy={sk_acc:.4f}, Time={sk_train_time:.2f}s")



gini_imp = rf_custom.feature_importances_

perm_imp = rf_custom.feature_importances_permutation(X_train, y_train, random_state=42)

plt.figure(figsize=(12, 6))
x = np.arange(len(feature_names))
width = 0.35
plt.bar(x - width/2, gini_imp, width, label='Gini importance', alpha=0.8)
plt.bar(x + width/2, perm_imp, width, label='Permutation importance (OOB)', alpha=0.8)
plt.xlabel('Признаки')
plt.ylabel('Важность')
plt.title('Сравнение важности признаков')
plt.xticks(x, feature_names, rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('images/importance_comparison.png', dpi=150)
plt.show()


print("\nИсследование зависимости ошибок от числа деревьев:")

n_estimators_range = range(1, 501, 5)
train_errors = []
test_errors = []
oob_errors = []

for n in n_estimators_range:
    rf = RandomForestCustom(
        n_estimators=n,
        max_features=best_max_features,
        max_depth=best_max_depth,
        oob_score=True,
        random_state=42,
        bootstrap=True
    )
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    train_err = 1 - accuracy_score(y_train, y_train_pred)
    train_errors.append(train_err)

    y_test_pred = rf.predict(X_test)
    test_err = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_err)

    oob_errors.append(rf.oob_score_)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_errors, label='Train error', linewidth=2)
plt.plot(n_estimators_range, test_errors, label='Test error', linewidth=2)
plt.plot(n_estimators_range, oob_errors, label='OOB error', linewidth=2, linestyle='--')
plt.xlabel('Число деревьев')
plt.ylabel('Ошибка')
plt.title('Зависимость ошибок от числа деревьев')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/errors_vs_n_estimators.png', dpi=150)
plt.show()
