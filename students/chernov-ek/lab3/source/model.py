import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm


RANDOM_STATE = 42
N_ESTIMATORS = 40
LEARNING_RATE = 0.1
MAX_DEPTH = 3


class SimpleGradientBoostingClassifier:
    """
    Простой градиентный бустинг для бинарной классификации.

    Attributes:
        n_estimators (int): Количество базовых деревьев. По умолчанию: 40.
        learning_rate (float): Шаг добавления каждого дерева. По умолчанию: 0.1.
        max_depth (int): Максимальная глубина базового дерева. По умолчанию: 3.
        min_samples_leaf (int): Минимум объектов в листе дерева. По умолчанию: 20.
        random_state (int): База для воспроизводимости деревьев. По умолчанию: 42.

    Fallbacks:
        При отсутствии положительного или отрицательного класса доля класса сглаживается.
    """

    def __init__(
        self,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    ):
        # Сохраняем гиперпараметры модели.
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        # Инициализируем обучаемые поля.
        self.initial_score = 0.0
        self.trees = []

    def _sigmoid(self, raw_scores):
        """
        Переводит сырые оценки модели в вероятности класса 1.

        Parameters:
            raw_scores (ndarray): Сырые оценки модели. По умолчанию: None.

        Returns:
            ndarray: Вероятности положительного класса.

        Fallbacks:
            Большие значения ограничиваются для защиты от переполнения экспоненты.
        """
        # Ограничиваем значения перед экспонентой.
        clipped_scores = np.clip(raw_scores, -30, 30)
        return 1.0 / (1.0 + np.exp(-clipped_scores))

    def fit(self, X, y):
        """
        Обучает ансамбль деревьев на антиградиентах логистической функции потерь.

        Parameters:
            X (ndarray): Матрица признаков. По умолчанию: None.
            y (ndarray): Целевой бинарный вектор. По умолчанию: None.

        Returns:
            SimpleGradientBoostingClassifier: Обученная модель.

        Fallbacks:
            Начальная вероятность сглаживается, чтобы избежать деления на ноль.
        """
        # Считаем стартовую константу как логит доли положительного класса.
        positive_rate = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.initial_score = np.log(positive_rate / (1 - positive_rate))

        # Начинаем прогноз с одной константы для всех объектов.
        raw_scores = np.full(y.shape[0], self.initial_score, dtype="float64")
        self.trees = []

        # Последовательно обучаем деревья на псевдоостатках с progress bar.
        progress_bar = tqdm(range(self.n_estimators), desc="Fit", unit="tree")
        for estimator_number in progress_bar:
            probabilities = self._sigmoid(raw_scores)
            residuals = y - probabilities

            # Строим простое регрессионное дерево как базовый алгоритм.
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + estimator_number,
            )
            tree.fit(X, residuals)

            # Добавляем вклад дерева к текущему ансамблю.
            raw_scores += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

            # Обновляем метрики progress bar после добавления дерева.
            updated_probabilities = np.clip(self._sigmoid(raw_scores), 1e-5, 1 - 1e-5)
            log_loss = -np.mean(
                y * np.log(updated_probabilities)
                + (1 - y) * np.log(1 - updated_probabilities)
            )
            residual_mae = np.mean(np.abs(y - updated_probabilities))
            progress_bar.set_postfix(
                {
                    "loss": f"{log_loss:.4f}",
                    "residual_mae": f"{residual_mae:.4f}",
                }
            )

        # Возвращаем модель в стиле sklearn.
        return self

    def decision_function(self, X):
        """
        Считает сырую оценку ансамбля до применения сигмоиды.

        Parameters:
            X (ndarray): Матрица признаков. По умолчанию: None.

        Returns:
            ndarray: Сырые оценки объектов.

        Fallbacks:
            Если деревьев нет, возвращается только начальная константа.
        """
        # Начинаем с базовой константы.
        raw_scores = np.full(X.shape[0], self.initial_score, dtype="float64")

        # Прибавляем прогнозы всех обученных деревьев.
        for tree in self.trees:
            raw_scores += self.learning_rate * tree.predict(X)

        return raw_scores

    def predict_proba(self, X):
        """
        Возвращает вероятности двух классов.

        Parameters:
            X (ndarray): Матрица признаков. По умолчанию: None.

        Returns:
            ndarray: Вероятности классов 0 и 1.

        Fallbacks:
            Вероятности строятся из текущей сырой оценки модели.
        """
        # Получаем вероятность положительного класса.
        positive_proba = self._sigmoid(self.decision_function(X))

        # Склеиваем вероятности классов в формат sklearn.
        return np.column_stack((1 - positive_proba, positive_proba))

    def predict(self, X):
        """
        Предсказывает бинарный класс по порогу 0.5.

        Parameters:
            X (ndarray): Матрица признаков. По умолчанию: None.

        Returns:
            ndarray: Предсказанные классы 0 или 1.

        Fallbacks:
            При ровной вероятности 0.5 выбирается класс 1.
        """
        # Превращаем вероятность в метку класса.
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
