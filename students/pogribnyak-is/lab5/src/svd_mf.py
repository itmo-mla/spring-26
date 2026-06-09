import numpy as np


class SVD:
    def __init__(self, n_factors: int = 50, n_epochs: int = 20,
                 lr: float = 0.005, reg: float = 0.02, seed: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.train_rmse_history: list[float] = []

    def fit(self, R: np.ndarray) -> "SVD":
        rng = np.random.default_rng(self.seed)
        n_users, n_items = R.shape

        self.mu = R[R > 0].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        obs = np.argwhere(R > 0)

        for epoch in range(self.n_epochs):
            rng.shuffle(obs)
            for u, i in obs:
                r = R[u, i]
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = r - pred

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])

            rmse = self._rmse_on_obs(R, obs)
            self.train_rmse_history.append(rmse)
            print(f"  Epoch {epoch + 1:>2}/{self.n_epochs}  train RMSE={rmse:.4f}")

        return self

    def _rmse_on_obs(self, R, obs):
        preds = self.mu + self.bu[obs[:, 0]] + self.bi[obs[:, 1]] + \
                np.einsum("ij,ij->i", self.P[obs[:, 0]], self.Q[obs[:, 1]])
        return float(np.sqrt(np.mean((R[obs[:, 0], obs[:, 1]] - preds) ** 2)))

    def predict_matrix(self) -> np.ndarray:
        return self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T

    def predict_rating(self, user_id: int, item_id: int) -> float:
        return float(self.mu + self.bu[user_id] + self.bi[item_id] +
                     self.P[user_id] @ self.Q[item_id])
