# Лабораторная работа №3. Градиентный бустинг

## Задание

1. Выбрать датасет для анализа.
2. Реализовать алгоритм градиентного бустинга.
3. Обучить модель на выбранном датасете.
4. Оценить качество модели с использованием кросс-валидации.
5. Замерить время обучения модели.
6. Сравнить результаты с эталонной реализацией из библиотеки `scikit-learn`.
7. Подготовить отчёт.

## Отчёт

### 1. Датасет

В качестве датасета выбран [Titanic Survival Prediction Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset).

Целевая переменная — **Survived** (1 — пассажир выжил, 0 — не выжил).

Подготовка данных:

```python
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = sns.load_dataset('titanic').dropna(subset=['age', 'embarked', 'fare'])
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['embarked'] = LabelEncoder().fit_transform(df['embarked'])

X = df.drop('survived', axis=1).values
y = df['survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Размер датасета: `712` объектов после очистки.


### 2. Программная реализация

Структура проекта:

```
lab3/
├── src/
│   └── GB.py      # Реализация градиентного бустинга
└── README.md
```

Реализован класс `GB` (`src/GB.py`), наследующий `BaseEstimator`, `ClassifierMixin` из sklearn.

Алгоритм на каждой итерации:
1. Считаем антиградиент log-loss: `residuals = y - sigmoid(Fm)`
2. Обучаем `DecisionTreeRegressor` на остатках
3. Ищем оптимальный `gamma` перебором по сетке `[0, 1]`
4. Обновляем ансамбль: `Fm += learning_rate * gamma * tree.predict(X)`

Начальное предсказание: `F0 = log(p / (1 - p))`, где `p = mean(y)`.

Параметры модели:

| Параметр | Значение |
|---|---|
| `n_estimators` | 100 |
| `learning_rate` | 0.1 |
| `max_depth` | 3 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `random_state` | 42 |

### 3. Анализ и сравнение

#### Кросс-валидация (5-fold)

```
GB (custom):   0.8189 ± 0.0230
GB (sklearn):  0.7992 ± 0.0308
```

#### Время обучения

```
GB (custom):   0.852 сек
GB (sklearn):  0.177 сек
```

#### Сравнение с эталоном

```python
GB(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
```

```
                  GB (custom)   sklearn
Accuracy            0.8189        0.7992
Std                 0.0230        0.0308
Время обучения      0.852 сек     0.177 сек
```

Собственная реализация показала более высокую точность при большем разбросе. sklearn обучается быстрее (~5×) видимо за счёт низкоуровневой реализации.

## Вывод

В ходе лабораторной работы реализован алгоритм градиентного бустинга для бинарной классификации. Модель обучена на датасете Titanic и достигла точности **81.9%** при кросс-валидации, превысив результат эталонной реализации sklearn (79.9%). По времени обучения sklearn быстрее примерно в 5 раз за счёт низкоуровневой реализации.
