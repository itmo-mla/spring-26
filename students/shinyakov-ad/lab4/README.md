# Лабораторная работа №4. EM-алгоритм

В работе реализован алгоритм Gaussian Mixture Model через EM-алгоритм и выполнено сравнение с `sklearn.mixture.GaussianMixture`.

## Датасет

Используется датасет [`dongeorge/seed-from-uci`](https://www.kaggle.com/datasets/dongeorge/seed-from-uci).

Файл данных: `Seed_Data.csv`.

Целевая переменная `target` используется только для внешней оценки кластеризации. В обучение GMM она не передается.

## Реализация

Собственная реализация находится в `source/model/model.py`.

Класс `GMM` поддерживает методы:

- `fit`
- `predict`
- `predict_proba`
- `score_samples`

## Оценка

Качество восстановления плотности оценивается через среднее лог-правдоподобие на тестовой выборке.

Для дополнительной оценки кластеризации используются:

- `accuracy`
- `AIC`
- `BIC`

## Запуск

```bash
python source/main.py
```

После запуска результаты сохраняются в папку `artifacts`.
