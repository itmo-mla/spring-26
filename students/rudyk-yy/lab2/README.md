# Лабораторная работа №2. Ансамбли моделей

В рамках лабораторной работы предстоит реализовать метод случайных подпространств (RSM) или Random Forest.

В качестве базовых алгоритмов рекомендуется использовать библиотечные реализации.

## Задание

1. Выбрать датасет для анализа;
2. Реализовать метод случайных подпространств (RSM) или Random Forest;
3. Обучить ансамбль, подобрать оптимальные гипер-параметры. Для подбора оптимальных параметров использовать grid search из sklearn; Оптимальные параметры подбирать по OOB;
4. Получить оценку важности признаков через OOB^j
5. Сравнить результаты с эталонными реализациями из библиотеки [scikit-learn](https://scikit-learn.org/stable/):
    * точность модели;
    * время обучения;
6. Подготовить отчет, включающий:
    * описание выбранного метода;
    * описание датасета;
    * результаты экспериментов;
    * сравнение с эталонными реализациями;
    * выводы.
## Отчёт

### 1. Датасет

Использован встроенный датасет Breast Cancer Wisconsin из sklearn:

- источник: `sklearn.datasets.load_breast_cancer()`;
- задача: бинарная классификация;
- train/test split: `test_size=0.3`, `random_state=42`.

### 2. Реализация метода

Реализация выполнена в `source/RandomForest.py`.

Что реализовано:

- bootstrap-выборка для каждого дерева - [code](source/RandomForest.py);
- случайный выбор подпространства признаков (`max_features`) ;
- ансамбль деревьев `DecisionTreeClassifier`;
- агрегация предсказаний голосованием;
- OOB-оценка качества (`oob_score`);
- OOB-важность признаков на основе permutation-подхода (`oob_feature_importance`).

### 3. Подбор гиперпараметров

Подбор выполнен в ноутбуке `lab2.ipynb` собственным grid search (вложенные циклы через `itertools.product`) по OOB-метрике.

Сетка гиперпараметров:

- `n_estimators`: `[10, 50, 100, 1000]`
- `max_depth`: `[None, 10, 20]`
- `max_features`: `['sqrt', 'log2']`
- `min_samples_split`: `[2, 5, 10]`

Результаты подбора:

- best params: `{'n_estimators': 100, 'max_depth': None, 'max_features': 'log2', 'min_samples_split': 10}`
- best OOB score: 0.9548
- test accuracy с лучшими параметрами: 0.9825
- total combinations: 72
#### Top 10 Combinations

| Rank | Score  | n_estimators | max_depth | max_features | min_samples_split |
|------|--------|--------------|-----------|--------------|-------------------|
| 1    | 0.9548 | 100          | None      | log2         | 10                |
| 2    | 0.9548 | 100          | 10        | log2         | 10                |
| 3    | 0.9548 | 100          | 20        | log2         | 10                |
| 4    | 0.9523 | 50           | None      | log2         | 5                 |
| 5    | 0.9523 | 50           | 10        | log2         | 5                 |
| 6    | 0.9523 | 50           | 20        | log2         | 5                 |
| 7    | 0.9523 | 1000         | None      | log2         | 2                 |
| 8    | 0.9523 | 1000         | 10        | log2         | 2                 |
| 9    | 0.9523 | 1000         | 20        | log2         | 2                 |
| 10   | 0.9497 | 50           | None      | log2         | 10                |

### 4. Важность признаков через OOB^j

Важность считалась как падение OOB-score после перестановки значений каждого признака.

Топ признаков:

| Признак                | OOB importance |
|------------------------|---------------|
| worst perimeter        | 0.6053        |
| worst texture          | 0.0921        |
| worst symmetry         | 0.0658        |
| mean concave points    | 0.0526        |
| worst radius           | 0.0526        |
| worst concave points   | 0.0395        |
| mean compactness       | 0.0263        |
| mean concavity         | 0.0263        |
| mean texture           | 0.0132        |
| mean smoothness        | 0.0132        |


### 5. Сравнение с sklearn

Сравнение проводилось с `sklearn.ensemble.RandomForestClassifier` на том же разбиении.

| Метрика                    | Моя реализация | sklearn |
|----------------------------|---------------|---------|
| Accuracy (test)            | 0.9708        | 0.9708  |
| Fit time (sec)             | 0.0579        | 0.0464  |
| Speed ratio (mine/sklearn) | 1.25x         | —       |

### 6. Файлы проекта

```
lab2/
├── source/
│   ├── RandomForest.py
│   └── SklearnRandomForest.py
├── lab2.ipynb
└── README.md
```

### Вывод

Реализован собственный Random Forest с OOB-оценкой и OOB-важностью признаков, проведён подбор гиперпараметров и сравнение с эталонной реализацией sklearn.



