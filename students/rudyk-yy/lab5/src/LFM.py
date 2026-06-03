import numpy as np


class LFM:
    """Latent Factor Model (Funk SVD) with SGD and user/item biases."""

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg_u=0.02, reg_i=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_u = reg_u
        self.reg_i = reg_i

    def fit(self, R):
        n_users, n_items = R.shape
        known = list(zip(*np.where(R > 0)))

        self.global_mean = R[R > 0].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        rng = np.random.default_rng(42)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        lr, reg_u, reg_i = self.lr, self.reg_u, self.reg_i

        for epoch in range(self.n_epochs):
            rng.shuffle(known)
            for u, i in known:
                err = R[u, i] - self._predict_scalar(u, i)
                self.bu[u] += lr * (err - reg_u * self.bu[u])
                self.bi[i] += lr * (err - reg_i * self.bi[i])
                pu, qi = self.P[u].copy(), self.Q[i].copy()
                self.P[u] += lr * (err * qi - reg_u * pu)
                self.Q[i] += lr * (err * pu - reg_i * qi)

        return self

    def _predict_scalar(self, u, i):
        return self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]

    def predict_all(self):
        R_hat = self.global_mean + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T
        return R_hat
