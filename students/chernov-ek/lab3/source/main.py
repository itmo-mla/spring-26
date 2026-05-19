from time import perf_counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data_loader import data_pipeline
from model import SimpleGradientBoostingClassifier


RANDOM_STATE = 42
N_SPLITS = 3
N_ESTIMATORS = 40
LEARNING_RATE = 0.1
MAX_DEPTH = 3


def build_custom_model():
    """
    Создает собственную модель градиентного бустинга.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        SimpleGradientBoostingClassifier: Новая модель для обучения.

    Fallbacks:
        Используются константы эксперимента из файла.
    """
    # Возвращаем новую модель для каждого фолда кросс-валидации.
    return SimpleGradientBoostingClassifier()


def build_sklearn_model():
    """
    Создает эталонную модель sklearn.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        GradientBoostingClassifier: Новая модель sklearn для обучения.

    Fallbacks:
        Используются гиперпараметры, близкие к собственной модели.
    """
    # Настраиваем sklearn-модель с тем же числом деревьев и глубиной.
    return GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model_builder, X, y):
    """
    Оценивает модель кросс-валидацией и измеряет время обучения.

    Parameters:
        model_builder (function): Функция создания новой модели. По умолчанию: None.
        X (ndarray): Матрица признаков. По умолчанию: None.
        y (ndarray): Целевой бинарный вектор. По умолчанию: None.

    Returns:
        dict: Средние значения ROC-AUC, accuracy и времени обучения.

    Fallbacks:
        Ошибки обучения или расчета метрик передаются вызывающему коду.
    """
    # Готовим стратифицированные разбиения для устойчивой оценки.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    accuracy_scores = []
    fit_times = []

    # Обучаем новую модель на каждом фолде.
    progress_bar = tqdm(cv.split(X, y), total=N_SPLITS, desc="CV", unit="fold")
    for train_index, test_index in progress_bar:
        model = model_builder()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Измеряем только время обучения модели.
        started_at = perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(perf_counter() - started_at)

        # Считаем качество по вероятностям и бинарным меткам.
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        auc_scores.append(roc_auc_score(y_test, y_proba))
        accuracy_scores.append(accuracy_score(y_test, y_pred))

        # Обновляем метрики в progress bar после каждого завершенного фолда.
        progress_bar.set_postfix(
            {
                "roc_auc": f"{np.mean(auc_scores):.4f}",
                "accuracy": f"{np.mean(accuracy_scores):.4f}",
                "fit_time": f"{np.mean(fit_times):.2f}s",
            }
        )

    # Собираем итоговые средние значения.
    return {
        "auc_mean": np.mean(auc_scores),
        "auc_std": np.std(auc_scores),
        "accuracy_mean": np.mean(accuracy_scores),
        "accuracy_std": np.std(accuracy_scores),
        "fit_time_mean": np.mean(fit_times),
    }


def print_results(results):
    """
    Печатает результаты эксперимента в виде таблицы.

    Parameters:
        results (dict): Словарь результатов по моделям. По умолчанию: None.

    Returns:
        None: Результаты выводятся в консоль.

    Fallbacks:
        Форматирование использует значения из переданного словаря.
    """
    # Печатаем заголовок таблицы.
    print("Модель | ROC-AUC | Accuracy | Время обучения, сек")
    print("-" * 58)

    # Печатаем строку результата для каждой модели.
    for model_name, metrics in results.items():
        auc_text = f"{metrics['auc_mean']:.4f} +/- {metrics['auc_std']:.4f}"
        accuracy_text = f"{metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}"
        time_text = f"{metrics['fit_time_mean']:.2f}"
        print(f"{model_name} | {auc_text} | {accuracy_text} | {time_text}")


def main():
    """
    Загружает данные, обучает две модели и сравнивает их качество.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        None: Итоговая таблица выводится в консоль.

    Fallbacks:
        Ошибки загрузки данных и обучения передаются вызывающему коду.
    """
    # Загружаем подготовленную матрицу признаков.
    X, y = data_pipeline()

    # Оцениваем собственную и эталонную реализации.
    results = {
        "Собственный бустинг": evaluate_model(build_custom_model, X, y),
        "sklearn GradientBoosting": evaluate_model(build_sklearn_model, X, y),
    }

    # Выводим сводную таблицу эксперимента.
    print_results(results)


# Запускаем сценарий только при прямом вызове файла.
if __name__ == "__main__":
    main()
