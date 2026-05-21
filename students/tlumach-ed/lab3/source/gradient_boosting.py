import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressorCustom:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.models = []
        self.gammas = []
        self.errors = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        # начальное приближение
        self.init_value = y.mean()
        self.F = np.full(n_samples, self.init_value)

        for _ in range(self.n_estimators):
            residual = y - self.F

            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residual)

            prediction = model.predict(X)

            gamma = np.sum(residual * prediction) / (np.sum(prediction ** 2) + 1e-8)

            self.F += self.learning_rate * gamma * prediction

            self.models.append(model)
            self.gammas.append(gamma)

            # лог ошибки
            mse = np.mean((y - self.F) ** 2)
            self.errors.append(mse)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_value)

        for model, gamma in zip(self.models, self.gammas):
            y_pred += self.learning_rate * gamma * model.predict(X)

        return y_pred