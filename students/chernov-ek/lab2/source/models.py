import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Класс пользовательской реализации Random Forest Classifier.

    Attributes:
        n_estimators (int): Количество деревьев. По умолчанию: 100.
        max_depth (int | None): Максимальная глубина дерева.
            По умолчанию: None.
        max_features (str | int): Количество признаков для разбиения.
            По умолчанию: "sqrt".
        random_state (int | None): Начальное значение генератора случайных
            чисел. По умолчанию: None.
        trees (list): Список обученных деревьев.
        oob_indices (list): Список OOB-индексов для каждого дерева.

    Fallbacks:
        Если max_features задан некорректно, используется
        полное количество признаков.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        random_state=None
    ):
        """
        Инициализация Random Forest Classifier.

        Parameters:
            n_estimators (int): Количество деревьев.
                По умолчанию: 100.
            max_depth (int | None): Максимальная глубина дерева.
                По умолчанию: None.
            max_features (str | int): Количество признаков для разбиения.
                По умолчанию: "sqrt".
            random_state (int | None): Seed генератора случайных чисел.
                По умолчанию: None.

        Returns:
            None: Конструктор ничего не возвращает.

        Fallbacks:
            Если max_features некорректен, используется
            полное количество признаков.
        """

        # Сохранение параметров модели
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        # Хранилище деревьев и OOB-индексов
        self.trees = []
        self.oob_indices = []

    def _get_max_features(self, n_features):
        """
        Вычисление количества признаков для дерева.

        Parameters:
            n_features (int): Общее количество признаков.

        Returns:
            int: Количество признаков для разбиения.

        Fallbacks:
            Если max_features указан некорректно,
            возвращается полное число признаков.
        """

        # Использование квадратного корня
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))

        # Использование логарифма по основанию 2
        if self.max_features == "log2":
            return int(np.log2(n_features))

        # Использование фиксированного количества признаков
        if isinstance(self.max_features, int):
            return self.max_features

        # Использование всех признаков
        return n_features

    def fit(self, X, y):
        """
        Обучение ансамбля деревьев решений.

        Parameters:
            X (numpy.ndarray): Матрица признаков.
            y (numpy.ndarray): Вектор классов.

        Returns:
            None: Метод ничего не возвращает.

        Fallbacks:
            Если OOB-объекты отсутствуют, дерево обучается
            только на bootstrap-выборке.
        """

        # Фиксация генератора случайных чисел
        np.random.seed(self.random_state)

        # Получение размеров выборки
        n_samples, n_features = X.shape

        # Определение количества признаков
        max_features = self._get_max_features(n_features)

        # Очистка старых деревьев
        self.trees = []
        self.oob_indices = []

        # Обучение каждого дерева
        for _ in range(self.n_estimators):

            # Bootstrap sampling
            bootstrap_indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True
            )

            # Получение OOB-индексов
            oob_idx = np.setdiff1d(
                np.arange(n_samples),
                bootstrap_indices
            )

            # Формирование bootstrap-выборки
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Создание дерева решений
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=max_features,
                random_state=np.random.randint(0, 100000)
            )

            # Обучение дерева
            tree.fit(X_bootstrap, y_bootstrap)

            # Сохранение дерева и OOB-индексов
            self.trees.append(tree)
            self.oob_indices.append(oob_idx)

    def predict(self, X):
        """
        Предсказание классов для объектов.

        Parameters:
            X (numpy.ndarray): Матрица признаков.

        Returns:
            numpy.ndarray: Предсказанные классы.

        Fallbacks:
            Если деревья отсутствуют, поведение sklearn
            может вызвать ошибку.
        """

        # Получение предсказаний всех деревьев
        predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        # Список итоговых предсказаний
        final_predictions = []

        # Голосование по каждому объекту
        for sample_predictions in predictions.T:

            # Выбор наиболее популярного класса
            most_common = Counter(
                sample_predictions
            ).most_common(1)[0][0]

            final_predictions.append(most_common)

        # Возврат итоговых предсказаний
        return np.array(final_predictions)

    def oob_score(self, X, y):
        """
        Вычисление OOB accuracy.

        Parameters:
            X (numpy.ndarray): Матрица признаков.
            y (numpy.ndarray): Истинные классы.

        Returns:
            float: Значение OOB accuracy.

        Fallbacks:
            Если объект не имеет OOB-предсказаний,
            он не участвует в оценке.
        """

        # Количество объектов
        n_samples = X.shape[0]

        # Список голосов для каждого объекта
        oob_votes = [[] for _ in range(n_samples)]

        # Получение OOB-предсказаний от деревьев
        for tree, oob_idx in zip(self.trees, self.oob_indices):

            # Пропуск пустых OOB-наборов
            if len(oob_idx) == 0:
                continue

            # Предсказания дерева
            preds = tree.predict(X[oob_idx])

            # Сохранение голосов
            for idx, pred in zip(oob_idx, preds):
                oob_votes[idx].append(pred)

        # Списки итоговых значений
        final_preds = []
        true_labels = []

        # Голосование по каждому объекту
        for i in range(n_samples):

            # Проверка наличия голосов
            if len(oob_votes[i]) > 0:

                # Получение наиболее частого класса
                vote = Counter(
                    oob_votes[i]
                ).most_common(1)[0][0]

                final_preds.append(vote)
                true_labels.append(y[i])

        # Преобразование в numpy-массивы
        final_preds = np.array(final_preds)
        true_labels = np.array(true_labels)

        # Вычисление accuracy
        accuracy = np.mean(final_preds == true_labels)

        return accuracy
