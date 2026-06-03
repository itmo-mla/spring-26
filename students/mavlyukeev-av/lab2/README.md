# Лабораторная работа №2.

## Описание метода

**Random Forest** — ансамблевый метод классификации. Строится множество деревьев решений; каждое обучается на bootstrap-выборке объектов, а при разбиении узла рассматривается случайное подмножество признаков (`max_features`).

Для каждого дерева $t$:
1. Формируется bootstrap-выборка $\mathcal{B}_t$ (выборка с возвращением размера $n$).
2. OOB-объекты — те, что не попали в $\mathcal{B}_t$ (в среднем ~36.8% выборки).
3. Дерево обучается только на $\mathcal{B}_t$.

**OOB-оценка** — accuracy по объектам, для которых есть предсказания хотя бы одного дерева, в котором объект был OOB. Используется для подбора гиперпараметров без отдельной валидационной выборки.

**Важность признаков OOB$^j$** — для каждого признака $j$ значения в OOB-предсказаниях случайно переставляются; важность = падение OOB-accuracy относительно базовой модели.

Реализация: `source/ensemble.py` (базовые деревья — `sklearn.tree.DecisionTreeClassifier`).

## Описание датасета

Использован встроенный датасет **Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`):
- **Задача**: бинарная классификация (доброкачественная / злокачественная опухоль).
- **Объектов**: 569, **признаков**: 30 (числовые, измерения ядра клеток).
- **Разбиение**: 80% train / 20% test, стратификация, `random_state=42`.
- **Пропусков нет**, предобработка не требуется.

## Подбор гиперпараметров

`GridSearchCV` из sklearn, scorer = OOB-accuracy, один «фиктивный» фолд (вся обучающая выборка).

Сетка параметров:
| Параметр | Возможные значения |
| --- | ---: |
| n_estimators | ```50, 100``` |
| max_depth    | ```null, 5, 10``` |
| min_samples_split | ```2, 5``` |
| max_features | ```sqrt, log2``` |

### Лучшие параметры
- Собственная реализация: `{"max_depth": null, "max_features": "sqrt", "min_samples_split": 5, "n_estimators": 50}`
- Sklearn `RandomForestClassifier`: `{"max_depth": 5, "max_features": "sqrt", "min_samples_split": 2, "n_estimators": 100}`

### OOB после подбора
| Модель | OOB accuracy |
|---|---:|
| Собственный RandomForest | 0.9626 |
| Sklearn RandomForestClassifier | 0.9626 |

## Результаты на тестовой выборке

| Модель | OOB | Accuracy | Precision | Recall | F1 | Время обучения (grid search), с |
|---|---:|---:|---:|---:|---:|---:|
| Собственный RandomForest | 0.9626 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 1.9841 |
| Sklearn RandomForestClassifier | 0.9626 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 2.1932 |

## Важность признаков (OOB$^j$)

| Признак | Важность |
|---|---:|
| worst concave points | 0.0220 |
| worst texture | 0.0176 |
| worst perimeter | 0.0176 |
| worst area | 0.0132 |
| mean smoothness | 0.0066 |
| mean texture | 0.0044 |
| texture error | 0.0044 |
| worst concavity | 0.0044 |
| worst symmetry | 0.0044 |
| mean perimeter | 0.0022 |
| mean area | 0.0022 |
| mean concavity | 0.0022 |
| mean fractal dimension | 0.0022 |
| radius error | 0.0022 |
| perimeter error | 0.0022 |
| concave points error | 0.0022 |
| symmetry error | 0.0022 |
| fractal dimension error | 0.0022 |

## Сравнение с эталоном sklearn

По **точности** (accuracy на test) собственная реализация совпадает с sklearn (0.9561 vs 0.9561), как и OOB-оценки (0.9626 vs 0.9626).

По **времени обучения** (полный grid search) собственная реализация заняла 1.98 с, sklearn — 2.19 с (отношение ~0.9×).

## Выводы

1. Реализован Random Forest с bootstrap, OOB-оценкой и пермутационной важностью признаков OOB$^j$.
2. Подбор гиперпараметров по OOB через `GridSearchCV` даёт сопоставимое качество с эталоном sklearn.
3. На датасете Breast Cancer обе модели показывают высокую точность (>95%).
4. Наиболее информативные признаки по OOB$^j$ — `worst concave points`, `worst texture`, `worst perimeter`, `worst area`.
5. Собственная реализация подходит для обучения и экспериментов, но для продакшена предпочтительнее `sklearn.ensemble.RandomForestClassifier` по скорости.
