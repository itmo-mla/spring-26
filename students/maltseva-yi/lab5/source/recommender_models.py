import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import Ridge
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter('ignore', SparseEfficiencyWarning)

class SLIM:
    def __init__(self, alpha=0.5, positive_only=True, max_iter=300, tol=1e-4):
        self.alpha = alpha
        self.positive_only = positive_only
        self.max_iter = max_iter
        self.tol = tol
        self.item_similarity = None

    def fit(self, X):
        X = X.tolil().copy()
        n_items = X.shape[1]
        self.item_similarity = np.zeros((n_items, n_items), dtype=np.float32)
        model = Ridge(alpha=self.alpha, fit_intercept=False, positive=self.positive_only,
                      max_iter=self.max_iter, tol=self.tol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for j in tqdm(range(n_items), desc="SLIM training"):
                y = X[:, j].toarray().ravel()
                X[:, j] = 0
                model.fit(X, y)
                self.item_similarity[:, j] = model.coef_
                self.item_similarity[j, j] = 0.0
                X[:, j] = y
        self.item_similarity = csr_matrix(self.item_similarity)
        return self

    def predict(self, X, user_indices=None):
        if user_indices is None:
            user_indices = np.arange(X.shape[0])
        pred = X[user_indices, :].dot(self.item_similarity).toarray()
        return np.clip(pred, 1.0, 5.0)

class ALS:
    def __init__(self, n_factors=50, n_epochs=20, reg=0.1, lr=0.015):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.lr = lr
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None

    def fit(self, X):
        X = X.tocsr()
        n_users, n_items = X.shape
        self.global_mean = X.data.mean()
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        rows, cols = X.nonzero()
        ratings = X.data
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for idx in range(len(ratings)):
                u, i = rows[idx], cols[idx]
                r = ratings[idx]
                pred = (self.global_mean + self.user_bias[u] + self.item_bias[i] +
                        np.dot(self.user_factors[u], self.item_factors[i]))
                err = r - pred
                total_loss += err * err
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])
                old_user = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * old_user - self.reg * self.item_factors[i])
            self.lr *= 0.95
            if epoch % 5 == 0:
                rmse = np.sqrt(total_loss / len(ratings))
                print(f"Epoch {epoch}, train RMSE = {rmse:.4f}, lr = {self.lr:.4f}")
        return self

    def predict_pair(self, user_indices, item_indices):
        preds = []
        for u, i in zip(user_indices, item_indices):
            p = (self.global_mean + self.user_bias[u] + self.item_bias[i] +
                 np.dot(self.user_factors[u], self.item_factors[i]))
            preds.append(np.clip(p, 1.0, 5.0))
        return np.array(preds)

    def predict_all(self):
        pred = (self.global_mean + self.user_bias[:, np.newaxis] +
                self.item_bias + np.dot(self.user_factors, self.item_factors.T))
        return np.clip(pred, 1.0, 5.0)