import numpy as np
from sklearn.linear_model import ElasticNet

class CustomSLIM:
    def __init__(self, alpha, l1_ratio, epochs, lr):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.epochs = epochs
        self.lr = lr
        self.W = None

    def fit(self, R):
        n_users, n_items = R.shape
        self.W = np.zeros((n_items, n_items))
        for epoch in range(self.epochs):
            pred = R @ self.W
            error = pred - R
            grad = (R.T @ error) / n_users + self.alpha * (self.l1_ratio * np.sign(self.W) + (1 - self.l1_ratio) * self.W)
            np.fill_diagonal(grad, 0)
            self.W -= self.lr * grad
            np.fill_diagonal(self.W, 0)
            self.W = np.maximum(self.W, 0)

    def predict(self, R):
        return R @ self.W

class ReferenceSLIM:
    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.W = None

    def fit(self, R):
        n_items = R.shape[1]
        self.W = np.zeros((n_items, n_items))
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, positive=True, fit_intercept=False)
        for i in range(n_items):
            X_train = np.delete(R, i, axis=1)
            y_train = R[:, i]
            if y_train.any():
                model.fit(X_train, y_train)
                self.W[:i, i] = model.coef_[:i]
                self.W[i+1:, i] = model.coef_[i:]

    def predict(self, R):
        return R @ self.W
