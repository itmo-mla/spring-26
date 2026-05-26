import time
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn import ensemble
from GradientBoosting import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from visuals import train_loss_plot, test_loss_plot

def calculate_metrics(y_true, y_pred):
    """
    Вычисляет основные метрики классификации

    Parameters:
    y_true : array-like, истинные метки классов
    y_pred : array-like, предсказанные метки классов
    """

    # Проверка на одинаковую длину массивов
    if len(y_true) != len(y_pred):
        raise ValueError("Массивы должны иметь одинаковую длину")

    # Вычисление метрик
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Вывод результатов
    print("МЕТРИКИ КЛАССИФИКАЦИИ")
    print("=" * 30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


data = load_breast_cancer()

X = data.data
y = data.target

print("Размер X:", X.shape)
print("Размер y:", y.shape)
print("Классы:", np.unique(y))

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

my_scores = []
my_times = []

sklearn_scores = []
sklearn_times = []

my_model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    )

sklearn_model = ensemble.GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    )


for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    start_time = time.time()
    my_model.fit(X_train, y_train)
    my_train_time = time.time() - start_time

    my_pred = my_model.predict(X_test)
    my_acc = accuracy_score(y_test, my_pred)
    my_pr = precision_score(y_test, my_pred)
    my_rc = recall_score(y_test, my_pred)
    my_f1 = f1_score(y_test, my_pred)

    my_scores.append((my_acc, my_pr, my_rc, my_f1))
    my_times.append(my_train_time)

    start_time = time.time()
    sklearn_model.fit(X_train, y_train)
    sklearn_train_time = time.time() - start_time

    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    sklearn_pr = precision_score(y_test, sklearn_pred)
    sklearn_rc = recall_score(y_test, sklearn_pred)
    sklearn_f1 = f1_score(y_test, sklearn_pred)

    sklearn_scores.append((sklearn_acc, sklearn_pr, sklearn_rc, sklearn_f1))
    sklearn_times.append(sklearn_train_time)

    print(f"Fold {fold}")
    print(f"Моя модель:     accuracy = {my_acc:.4f}, time = {my_train_time:.4f} сек")
    print(f"Sklearn модель: accuracy = {sklearn_acc:.4f}, time = {sklearn_train_time:.4f} сек")
    print("-" * 60)

my_scores = np.array(my_scores)
sklearn_scores = np.array(sklearn_scores)

metrics_names = ["Accuracy", "Precision", "Recall", "F1"]

print("\nМоя реализация:")
for i, metric_name in enumerate(metrics_names):
    print(f"{metric_name}: mean = {my_scores[:, i].mean():.4f}")

print("\nSklearn реализация:")
for i, metric_name in enumerate(metrics_names):
    print(f"{metric_name}: mean = {sklearn_scores[:, i].mean():.4f}")

my_model.fit(X_train, y_train)
train_loss_plot(my_model, X_train, y_train)
test_loss_plot(my_model, X_test, y_test)

