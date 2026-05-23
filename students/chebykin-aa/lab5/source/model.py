import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import TruncatedSVD


class SLIM:
    def __init__(self, reg: float = 200.0):
        self.reg = reg
        self.W_: np.ndarray | None = None

    def fit(self, R: np.ndarray):
        G = R.T @ R
        d = np.arange(G.shape[0])
        G[d, d] += self.reg
        P = np.linalg.inv(G)
        W = -P / np.diag(P)[np.newaxis, :]
        W[d, d] = 0.0
        self.W_ = W
        return self

    def predict_matrix(self, R: np.ndarray) -> np.ndarray:
        return R @ self.W_


class SLIMRef:
    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, max_iter: int = 200):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.W_: np.ndarray | None = None

    def fit(self, R: np.ndarray):
        n_items = R.shape[1]
        self.W_ = np.zeros((n_items, n_items))
        enet = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            positive=True,
            fit_intercept=False,
        )
        for j in range(n_items):
            y = R[:, j].copy()
            if not y.any():
                continue
            r_j = R[:, j].copy()
            R[:, j] = 0.0
            enet.fit(R, y)
            R[:, j] = r_j
            self.W_[:, j] = enet.coef_
        return self

    def predict_matrix(self, R: np.ndarray) -> np.ndarray:
        return R @ self.W_


class SVD:
    def __init__(
        self,
        n_factors: int = 20,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        random_state: int | None = None,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        # заполняются после fit()
        self.U_: np.ndarray | None = None
        self.V_: np.ndarray | None = None
        self.bu_: np.ndarray | None = None
        self.bi_: np.ndarray | None = None
        self.mu_: float = 0.0

    def fit(self, R: np.ndarray) -> "SVD":
        rng = np.random.RandomState(self.random_state)
        n_users, n_items = R.shape
        mask = R > 0
        self.mu_ = float(R[mask].mean())
        self.U_ = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.V_ = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.bu_ = np.zeros(n_users)
        self.bi_ = np.zeros(n_items)

        users, items = np.where(mask)
        idx = np.arange(len(users))
        for _ in range(self.n_epochs):
            rng.shuffle(idx)
            for k in idx:
                u, i = users[k], items[k]
                err = R[u, i] - (self.mu_ + self.bu_[u] + self.bi_[i] + self.U_[u] @ self.V_[i])
                self.bu_[u] += self.lr * (err - self.reg * self.bu_[u])
                self.bi_[i] += self.lr * (err - self.reg * self.bi_[i])
                u_old = self.U_[u].copy()
                self.U_[u] += self.lr * (err * self.V_[i] - self.reg * self.U_[u])
                self.V_[i] += self.lr * (err * u_old - self.reg * self.V_[i])
        return self

    def predict_matrix(self) -> np.ndarray:
        return self.mu_ + self.bu_[:, np.newaxis] + self.bi_[np.newaxis, :] + self.U_ @ self.V_.T


class SVDRef:
    def __init__(self, n_factors: int = 20, random_state: int | None = None):
        self.n_factors = n_factors
        self.random_state = random_state
        self._R_pred: np.ndarray | None = None

    def fit(self, R: np.ndarray) -> "SVDRef":
        tsvd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        U = tsvd.fit_transform(R)
        self._R_pred = U @ tsvd.components_
        return self

    def predict_matrix(self) -> np.ndarray:
        return self._R_pred
