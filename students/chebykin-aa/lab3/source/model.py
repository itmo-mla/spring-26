import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Заполняются после fit()
        self.estimators_: list[DecisionTreeRegressor] = []
        self.init_prediction_: float = 0.0
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Ограничиваем, чтобы избежать переполнения
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.estimators_ = []

        # Начальное предсказание: log-odds положительного класса
        pos_ratio = y.mean()
        self.init_prediction_ = np.log(pos_ratio / (1.0 - pos_ratio))

        # Текущие предсказания в логитах
        F = np.full(X.shape[0], self.init_prediction_)

        for _ in range(self.n_estimators):
            # Вероятности текущей модели
            p = self._sigmoid(F)

            # Антиградиент log-loss: y - p
            residuals = y - p

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.rng.randint(0, 2**31),
            )
            tree.fit(X, residuals)
            self.estimators_.append(tree)

            # Обновляем предсказания
            F += self.learning_rate * tree.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full(X.shape[0], self.init_prediction_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        p = self._sigmoid(F)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
