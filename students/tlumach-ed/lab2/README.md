# Лабораторная работа №2: Ансамбли моделей (Random Forest)

## 1. Датасет

Использован датасет **Adult (UCI / OpenML)** — задача бинарной классификации:
- цель: предсказать, превышает ли доход 50K
- объекты: ~48k
- признаки: числовые + категориальные (one-hot encoding)
- присутствует **дисбаланс классов** (~75% / 25%)
- категориальные признаки преобразованы через one-hot encoding

---

## 2. Метод

Реализован ансамбль на основе **Random Forest**.

- **Bagging (bootstrap)** — обучение деревьев на подвыборках с возвращением
- **Random Subspaces** — случайный выбор подмножества признаков
- **Ансамблирование** — объединение предсказаний

---

## 3. Подбор гиперпараметров (OOB)

Использован кастомный Grid Search с метрикой **OOB score**.

Для подбора гиперпараметров был реализован кастомный grid search, так как стандартный GridSearchCV из sklearn не поддерживает использование OOB-оценки в качестве метрики оптимизации.

Это связано с тем, что GridSearchCV требует функцию качества вида scoring(estimator, X, y) и выполняет кросс-валидацию, разбивая данные на обучающие и валидационные подвыборки. В то же время OOB-оценка:

- вычисляется внутри метода fit,
- не требует отдельного валидационного множества,
- доступна только как атрибут обученной модели (oob_score_).

issue - https://github.com/scikit-learn/scikit-learn/issues/23382

Результаты:

n_trees=100, depth=12, min_split=5 → OOB=0.8533 (best)

Лучшие параметры:
- n_trees = 100
- max_depth = 12
- min_samples_split = 5

---

## 4. OOB (Out-of-Bag)

OOB используется как:
- оценка качества без валидации
- критерий подбора гиперпараметров
- база для feature importance

Идея:
каждый объект предсказывается только теми деревьями, которые не видели его при обучении.


---

## 5. Feature Importance (OOB^j)

Использована пермутационная важность:

importance_j = (OOB - OOB_j) / OOB * 100%

где:
- OOB — базовое качество
- OOB_j — качество после перемешивания признака j

![график важности признаков](\students\tlumach-ed\lab2\source\Figure_1.png)

---

## 6. Результаты модели

### Custom Random Forest

- accuracy: 0.867
- precision: 0.797
- recall: 0.574
- f1: 0.667
- OOB: 0.853
- train_time: 590 сек

---


## 7. Сравнение со sklearn

### Sklearn RandomForestClassifier

- accuracy: 0.8666  
- precision: 0.7984  
- recall: 0.5731  
- f1: 0.6672  
- train_time: 1.28 сек  

---

### Сравнение

| Метрика   | Custom RF | Sklearn RF |
|----------|-----------|-----------|
| Accuracy | 0.867     | 0.867     |
| Precision| 0.797     | 0.798     |
| Recall   | 0.574     | 0.573     |
| F1-score | 0.667     | 0.667     |
| Время    | 590 сек   | 1.28 сек  |

---

## 8. OOB Curve

Построен график зависимости OOB error от числа деревьев.


![график](\students\tlumach-ed\lab2\source\Figure_2.png)

---

## 9. Выводы

- Реализован Random Forest с использованием bagging и случайных подпространств
- OOB используется для:
  - оценки качества
  - подбора гиперпараметров
  - оценки важности признаков
- Модель показала сопоставимое качество со sklearn





```
=== Custom GridSearch (OOB) ===
n_trees=50, depth=6, min_split=5 → OOB=0.8290
n_trees=50, depth=6, min_split=10 → OOB=0.8261
n_trees=50, depth=10, min_split=5 → OOB=0.8462
n_trees=50, depth=10, min_split=10 → OOB=0.8479
n_trees=50, depth=12, min_split=5 → OOB=0.8511
n_trees=50, depth=12, min_split=10 → OOB=0.8509
n_trees=100, depth=6, min_split=5 → OOB=0.8280
n_trees=100, depth=6, min_split=10 → OOB=0.8297
n_trees=100, depth=10, min_split=5 → OOB=0.8487
n_trees=100, depth=10, min_split=10 → OOB=0.8481
n_trees=100, depth=12, min_split=5 → OOB=0.8533
n_trees=100, depth=12, min_split=10 → OOB=0.8523

Best params: (100, 12, 5)
Best OOB score: 0.853315393840124

=== Final Model ===
{'accuracy': 0.8668532041220228, 'precision': 0.8026370004120313, 'recall': 0.5695906432748538, 'f1': 0.6663246109115786}
OOB score: 0.8532276463189915
Train time: 590.3409249782562

Feature importance (first 10): [ 0.4250797   0.0274245   2.12197045  4.29536183  0.61362312  0.37708683
  0.03085256 -0.01028419  0.          0.09255768]

```




