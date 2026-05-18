import numpy as np


def calculate_gini_impurity(labels: np.ndarray) -> float:
    """
    Вычисляет неоднородность Джини для набора меток.

    Parameters:
        labels (numpy.ndarray): Массив меток классов. По умолчанию: нет.

    Returns:
        float: Значение неоднородности Джини.

    Fallbacks:
        Для пустого массива возвращается 0.
    """
    # Пустой узел не содержит неопределенности.
    if len(labels) == 0:
        return 0

    # Считаем долю каждого класса в переданном наборе.
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    return 1 - np.sum(probabilities**2)


def calculate_weighted_gini_impurity(
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Вычисляет неоднородность Джини для меток с учётом весов объектов.

    Parameters:
        labels (numpy.ndarray): Массив меток классов. По умолчанию: нет.
        weights (numpy.ndarray): Веса объектов. По умолчанию: нет.

    Returns:
        float: Значение неоднородности Джини.

    Fallbacks:
        Для пустого массива или нулевой суммы весов возвращается 0.
    """
    # Пустой набор или нулевой вес не создают неоднородности.
    if len(labels) == 0 or np.sum(weights) == 0:
        return 0

    # Суммируем веса отдельно по каждому классу.
    unique_labels = np.unique(labels)
    class_weights = np.array(
        [np.sum(weights[labels == label]) for label in unique_labels],
        dtype=float,
    )
    probabilities = class_weights / np.sum(class_weights)

    return 1 - np.sum(probabilities**2)


def is_numeric_column(values: np.ndarray) -> bool:
    """
    Проверяет, можно ли привести значения признака к числовому типу.

    Parameters:
        values (numpy.ndarray): Значения одного признака. По умолчанию: нет.

    Returns:
        bool: True, если признак числовой, иначе False.

    Fallbacks:
        При ошибке приведения типа возвращается False.
    """
    # Пробное приведение отделяет количественные признаки от категориальных.
    try:
        values.astype(float)
    except (ValueError, TypeError):
        return False

    return True


def calculate_gini_from_counts(counts: np.ndarray) -> float:
    """
    Вычисляет неоднородность Джини по количествам классов.

    Parameters:
        counts (numpy.ndarray): Количество объектов каждого класса. По умолчанию: нет.

    Returns:
        float: Значение неоднородности Джини.

    Fallbacks:
        Если суммарное количество объектов равно 0, возвращается 0.
    """
    # Пустое разбиение не влияет на итоговую оценку.
    total = counts.sum()
    if total == 0:
        return 0

    # Переводим количества в вероятности классов.
    probabilities = counts / total

    return 1 - np.sum(probabilities**2)


def calculate_numeric_information_gain(
    feature_values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float | None]:
    """
    Находит лучший порог и прирост информации для числового признака.

    Parameters:
        feature_values (numpy.ndarray): Значения числового признака. По умолчанию: нет.
        labels (numpy.ndarray): Метки классов для объектов. По умолчанию: нет.
        weights (numpy.ndarray): Веса объектов. По умолчанию: нет.

    Returns:
        tuple[float, float | None]: Лучший прирост информации и найденный порог.

    Fallbacks:
        Если признак не дает разбиения, возвращаются 0 и None.
    """
    # Сортируем объекты по значению признака для перебора соседних порогов.
    numeric_values = feature_values.astype(float)
    order = np.argsort(numeric_values)
    sorted_values = numeric_values[order]
    sorted_labels = labels[order]
    sorted_weights = weights[order]

    # Один уникальный уровень признака не может улучшить дерево.
    if len(np.unique(sorted_values)) <= 1:
        return 0, None

    # Кодируем классы в индексы для быстрых векторных подсчетов.
    unique_labels, encoded_labels = np.unique(sorted_labels, return_inverse=True)
    total_counts = np.zeros(len(unique_labels), dtype=float)
    np.add.at(total_counts, encoded_labels, sorted_weights)
    left_counts = np.zeros_like(total_counts)
    total_weight = np.sum(sorted_weights)
    parent_gini = calculate_gini_from_counts(total_counts)

    # Инициализируем лучший найденный порог.
    best_gain = 0
    best_threshold = None

    # Перебираем границы между разными соседними значениями признака.
    for index in range(len(sorted_values) - 1):
        left_counts[encoded_labels[index]] += sorted_weights[index]

        if sorted_values[index] == sorted_values[index + 1]:
            continue

        # Считаем взвешенную неоднородность двух дочерних узлов.
        right_counts = total_counts - left_counts
        left_weight = left_counts.sum() / total_weight
        right_weight = right_counts.sum() / total_weight
        weighted_child_gini = (
            left_weight * calculate_gini_from_counts(left_counts)
            + right_weight * calculate_gini_from_counts(right_counts)
        )
        gain = parent_gini - weighted_child_gini

        # Обновляем лучший порог при улучшении прироста информации.
        if gain > best_gain:
            best_gain = gain
            best_threshold = (sorted_values[index] + sorted_values[index + 1]) / 2

    return best_gain, best_threshold


def calculate_information_gain(
    features: np.ndarray,
    labels: np.ndarray,
    feature_index: int,
    weights: np.ndarray,
    missing_values: set[str] | None = None,
) -> tuple[float, float | None, bool]:
    """
    Вычисляет прирост информации для выбранного признака.

    Parameters:
        features (numpy.ndarray): Матрица признаков. По умолчанию: нет.
        labels (numpy.ndarray): Метки классов. По умолчанию: нет.
        feature_index (int): Индекс проверяемого признака. По умолчанию: нет.
        weights (numpy.ndarray): Веса объектов. По умолчанию: нет.
        missing_values (set[str] | None): Маркеры пропущенных значений. По умолчанию: None.

    Returns:
        tuple[float, float | None, bool]: Прирост, порог и флаг числового признака.

    Fallbacks:
        Для категориальных признаков порог возвращается как None.
    """
    # Извлекаем столбец признака из матрицы объектов.
    feature_values = features[:, feature_index]
    weights = np.array(weights, dtype=float)
    missing_values = set() if missing_values is None else missing_values

    # Для числового признака ищем бинарное разбиение по порогу.
    if is_numeric_column(feature_values):
        gain, threshold = calculate_numeric_information_gain(
            feature_values,
            labels,
            weights,
        )
        return gain, threshold, True

    # Исключаем пропуски из оценки категориального разбиения.
    observed_mask = np.ones(len(feature_values), dtype=bool)
    if missing_values:
        observed_mask = ~np.isin(feature_values, list(missing_values))

    # Если признак целиком состоит из пропусков, разбиение невозможно.
    if not np.any(observed_mask):
        return 0, None, False

    # Оцениваем разбиение только по наблюдаемым значениям и штрафуем за пропуски.
    observed_values = feature_values[observed_mask]
    observed_labels = labels[observed_mask]
    observed_weights = weights[observed_mask]
    observed_total_weight = np.sum(observed_weights)
    total_weight = np.sum(weights)
    parent_gini = calculate_weighted_gini_impurity(observed_labels, observed_weights)
    values = np.unique(observed_values)
    weighted_child_gini = 0

    # Суммируем вклад каждого дочернего узла.
    for value in values:
        subset_mask = observed_values == value
        subset_labels = observed_labels[subset_mask]
        subset_weights = observed_weights[subset_mask]
        weighted_child_gini += (
            np.sum(subset_weights) / observed_total_weight
        ) * calculate_weighted_gini_impurity(subset_labels, subset_weights)

    return (
        (observed_total_weight / total_weight) * (parent_gini - weighted_child_gini),
        None,
        False,
    )
