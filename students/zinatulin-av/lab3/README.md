# Лабораторная работа 3. Градиентный бустинг

Реализация и сравнение собственного `GradientBoostingClf` с эталоном из `scikit-learn`.

## Описание алгоритма

Градиентный бустинг строит ансамбль базовых алгоритмов последовательно: каждый следующий пытается исправить ошибки предыдущих, приближая антиградиент функции потерь.

Схема обучения:
1. Инициализация: $a_0(x) = 0$.
2. На каждой итерации $t = 1, \dots, T$:
    - считаем антиградиент в текущих точках: $g_i = -L'(a_{t-1}(x_i),\, y_i)$;
    - обучаем базовый алгоритм $b_t$ приближать вектор $(g_i)$:
    $b_t = \arg\min_b \sum_i (b(x_i) - g_i)^2$;
    - находим шаг одномерной минимизацией: $\alpha_t = \arg\min_{\alpha > 0} \sum_i L(a_{t-1}(x_i) + \alpha b_t(x_i),\, y_i)$;
    - обновляем приближение: $a_t(x) = a_{t-1}(x) + \alpha_t\, b_t(x)$.
3. Итоговый ансамбль: $a(x) = \sum_{t=1}^T \alpha_t b_t(x)$.

В реализации:
- базовый алгоритм — `DecisionTreeRegressor`;
- функция потерь — логистическая $L(a, y) = \log(1 + e^{-ay})$, $y \in \{-1, +1\}$;
- шаг $\alpha_t$ ищется методом `scipy.optimize.minimize_scalar`.

## Описание датасета

**Wine Quality (Red)** — данные о красном португальском вине Vinho Verde.

- 1599 объектов;
- 11 числовых признаков физико-химических свойств (кислотность, сахар, алкоголь и т.п.);
- целевая переменная — экспертная оценка качества от 3 до 8.

Для бинарной классификации задача переформулирована: класс 1 — качество $\geq 6$ (хорошее вино), иначе класс 0.

```python
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from source.GradientBoostingClf import GradientBoostingClf
```

```python
df = pd.read_csv('dataset/winequality-red.csv')
X = df.drop(columns=['quality']).values
y = (df['quality'].values >= 6).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'Размер выборки: {X.shape[0]} объектов, {X.shape[1]} признаков')
print(f'Доля класса 1: {y.mean():.3f}')
```

```
Размер выборки: 1599 объектов, 11 признаков
Доля класса 1: 0.535
```

## Эксперименты

Сравниваем две модели с одинаковыми гиперпараметрами: `n_estimators=500`, `max_depth=3`.

```python
params = dict(n_estimators=500, max_depth=3)

my_clf = GradientBoostingClf(**params)
start = time.perf_counter()
my_clf.fit(X_train, y_train)
my_time = time.perf_counter() - start
my_acc = accuracy_score(y_test, my_clf.predict(X_test))

sk_clf = GradientBoostingClassifier(random_state=42, **params)
start = time.perf_counter()
sk_clf.fit(X_train, y_train)
sk_time = time.perf_counter() - start
sk_acc = accuracy_score(y_test, sk_clf.predict(X_test))

print(f'My GB     : accuracy = {my_acc:.4f}, time = {my_time:.3f} с')
print(f'Sklearn GB: accuracy = {sk_acc:.4f}, time = {sk_time:.3f} с')
```

```
My GB     : accuracy = 0.7875, time = 0.851 с
Sklearn GB: accuracy = 0.8063, time = 0.607 с
```

```python
my_cv = cross_val_score(GradientBoostingClf(**params), X, y, cv=5, scoring='accuracy')
sk_cv = cross_val_score(GradientBoostingClassifier(random_state=42, **params), X, y, cv=5, scoring='accuracy')

print(f'My GB     : CV accuracy = {my_cv.mean():.4f} ± {my_cv.std():.4f}')
print(f'Sklearn GB: CV accuracy = {sk_cv.mean():.4f} ± {sk_cv.std():.4f}')
```

```
My GB     : CV accuracy = 0.6735 ± 0.0274
Sklearn GB: CV accuracy = 0.6992 ± 0.0215
```

```python
results = pd.DataFrame({
    'Test accuracy': [my_acc, sk_acc],
    'CV accuracy (mean)': [my_cv.mean(), sk_cv.mean()],
    'CV accuracy (std)': [my_cv.std(), sk_cv.std()],
    'Train time, с': [my_time, sk_time],
}, index=['My GradientBoostingClf', 'Sklearn GradientBoostingClassifier'])
results.round(4)
```

|                                     | Test accuracy | CV accuracy (mean) | CV accuracy (std) | Train time, с |
| ----------------------------------- | ------------: | -----------------: | ----------------: | ------------: |
| My GradientBoostingClf              |        0.7875 |             0.6735 |            0.0274 |        0.8509 |
| Sklearn GradientBoostingClassifier  |        0.8063 |             0.6992 |            0.0215 |        0.6068 |

## Выводы

- На отложенной выборке точности близкие: 0.788 у собственной реализации против 0.806 у `scikit-learn`.
- На 5-fold кросс-валидации `scikit-learn` немного лучше: 0.699 против 0.674. Главная причина — в собственной реализации нет learning rate, полный шаг $\alpha_t$ от линейного поиска быстро приводит к переобучению.
- Время обучения сравнимое: 0.85 с против 0.61 с. На больших выборках `scikit-learn` будет значительно быстрее за счёт C-реализации деревьев.
