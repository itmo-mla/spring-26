import numpy as np
from itertools import product

from models import RandomForestClassifier as CustomRandomForestClassifier


def grid_search_oob(X, y, param_grid, model_class) -> tuple[dict, CustomRandomForestClassifier]:
    """
    Подбор гиперпараметров модели по OOB-оценке.

    Parameters:
        X (np.ndarray): Матрица признаков.
        y (np.ndarray): Вектор классов.
        param_grid (dict): Сетка параметров.
        model_class (class): Класс модели Random Forest.

    Returns:
        dict: Лучшие параметры.
        object: Лучшая обученная модель.

    Fallbacks:
        Если OOB-score не может быть вычислен,
        модель считается невалидной.
    """

    # Формируем все комбинации параметров
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score = -np.inf
    best_params = None
    best_model = None

    # Перебор всех комбинаций (аналог grid search)
    for params in product(*values):

        param_dict = dict(zip(keys, params))

        # Создание модели с текущими параметрами
        model = model_class(**param_dict)

        # Обучение модели
        model.fit(X, y)

        # OOB оценка качества
        score = model.oob_score(X, y)

        # Сохранение лучшей модели
        if score > best_score:
            best_score = score
            best_params = param_dict
            best_model = model

        print(f"params={param_dict}, oob_score={score:.4f}")

    return best_params, best_model
