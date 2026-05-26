import time
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from scipy.optimize import minimize_scalar

class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.models = []
        self.alphas = []

    def _loss(self, a, y):
        return np.logaddexp(0, -y * a)

    def _loss_derivative(self, a, y):
        return -y / (1 + np.exp(y * a))

    def fit(self, X, y):
        y = np.where(y == 1, 1, -1)
        n_samples = X.shape[0]

        a = np.zeros(n_samples)

        self.models = []
        self.alphas = []

        for t in range(self.n_estimators):
            antigradient = -self._loss_derivative(a, y)
            base_model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + t
            )

            base_model.fit(X, antigradient)
            b_pred = base_model.predict(X)

            def objective(alpha):
                new_a = a + alpha * b_pred
                return np.sum(self._loss(new_a, y))

            result = minimize_scalar(
                objective,
                bounds=(0, 10),
                method="bounded"
            )

            alpha = result.x
            a = a + alpha * b_pred

            self.models.append(base_model)
            self.alphas.append(alpha)

        return self

    def decision_function(self, X):
        result = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            result += alpha * model.predict(X)

        return result

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)
    
    def _partial_predict(self, df, k: int, y_true):
        y_score = np.zeros(df.shape[0])

        for i in range(k):
            pred = self.models[i].predict(df)
            y_score += self.alphas[i] * pred

        y_pred = np.sign(y_score)

        y_pred[y_pred == 0] = 1

        return np.mean(y_pred != y_true)