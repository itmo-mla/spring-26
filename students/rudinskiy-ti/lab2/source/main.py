import kagglehub
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

from RandomForestClassifier import RandomForestClassifier
from visuals import test_loss_plot, train_loss_plot

EPS = 1e-5
MAX_ITERS = 1e5

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

def drop_outliers(df:pd.DataFrame, columns:list):
    """
    Функция для удаления выбросов.
    На вход принимает дасатет и список колонок для очистки.
    Возвращает очищенный от выбросов датасет.
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q2 = df[column].quantile(0.75)
        IQR = Q2 - Q1
        df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q2 + 1.5 * IQR)]

    return df

def column_normalisation(df:pd.DataFrame, columns:list):

    for column in columns:
        median = df[column].median()
        IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
        IQR = IQR if IQR != 0 else 1e-6

        df[column] = df[column].apply(lambda x: (x - median)/IQR)

    return df

def oob_scorer(estimator, X, y):
    """
    Scorer для GridSearchCV.
    estimator - обученная модель RandomForestClassifier
    X, y - данные, на которых она обучена
    """
    return - estimator.out_of_bag_score(X, y)

dataset_path = kagglehub.dataset_download("muhammedderric/fitness-classification-dataset-synthetic")
csv_path = os.path.join(dataset_path, 'fitness_dataset.csv')
df = pd.read_csv(csv_path)
# df['bias'] = 1
df = drop_outliers(df, ['heart_rate'])
df = column_normalisation(df, ['age', 'height_cm', 'weight_kg', 'heart_rate', 'blood_pressure', 'sleep_hours', 'nutrition_quality', 'activity_index'])
replace_vals = {
    'smokes': {
        'no': -1,
        'yes': 1,
        '0': -1,
        '1': 1
    },
    'gender': {
        'F': -1,
        'M': 1
    },
    'is_fit': {
        0: -1
    }
}
df['sleep_hours'].fillna(df['sleep_hours'].median(), inplace=True)
df = df.replace(replace_vals)

Y = df['is_fit']
X = df.drop(['is_fit', 'blood_pressure', 'sleep_hours'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(1000, 7, 40, 5, 5, 5, criterion='log_loss')
model.fit(X_train, y_train)
rs = model.predict(X_test)
res = calculate_metrics(y_test, rs)
train_loss_plot(model, X_train, y_train)
test_loss_plot(model, X_test, y_test)

sk_model = ensemble.RandomForestClassifier(
    criterion='log_loss',            
    n_estimators=400,            
    max_samples=1000,           
    max_features=8,            
    random_state=42            
)
sk_model.fit(X_train, y_train)
res = sk_model.predict(X_test)
print(calculate_metrics(y_test, res))


for i in X_train.columns:
    print(f'{i}: {model.importance(i, X_train, y_train):.2f}')

param_grid = {
    'n_estimators': [200, 400],
    'l_sample': [500, 1000],
    'n_feat': [5, 8],
    'tol_train': [5],
    'tol_valid': [5],
    'base_classifier__max_depth': [3, 5],
    'base_classifier__criterion': ['gini', 'log_loss']
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=oob_scorer,
    verbose=2,      
    error_score='raise',
    n_jobs=1
)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)
print("Минимальная OOB ошибка:", grid_search.best_score_)