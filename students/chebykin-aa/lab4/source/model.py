import numpy as np


class GaussianMixture:
    def __init__(
        self,
        n_components: int = 1,
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        # заполняются после fit()
        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0

    def _log_gauss(self, X: np.ndarray, k: int) -> np.ndarray:
        # log N(x | mu_k, diag(var_k)) для каждого объекта
        var = self.variances_[k] + self.reg_covar
        d = X.shape[1]
        diff = X - self.means_[k]
        log_det = np.sum(np.log(var))
        maha = np.sum(diff ** 2 / var, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

    def _init_kmeans_pp(self, X: np.ndarray, rng: np.random.RandomState):
        n, d = X.shape
        K = self.n_components

        # выбираем первый центр случайно, остальные — с вероятностью пропорциональной расстоянию
        idx = rng.randint(n)
        centers = [X[idx].copy()]
        for _ in range(1, K):
            dists = np.min(
                np.array([np.sum((X - c) ** 2, axis=1) for c in centers]),
                axis=0,
            )
            probs = np.maximum(dists, 0) / (np.maximum(dists, 0).sum() + 1e-300)
            centers.append(X[rng.choice(n, p=probs)].copy())

        self.means_ = np.array(centers)

        # начальные дисперсии по кластерам ближайшего центра
        assignments = np.argmin(
            np.array([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0
        )
        self.variances_ = np.ones((K, d))
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 1:
                self.variances_[k] = np.var(X[mask], axis=0) + self.reg_covar

        self.weights_ = np.full(K, 1.0 / K)

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        n, K = X.shape[0], self.n_components
        log_ll = np.empty((n, K))
        for k in range(K):
            log_ll[:, k] = np.log(self.weights_[k] + 1e-300) + self._log_gauss(X, k)

        log_sum = np.logaddexp.reduce(log_ll, axis=1)
        log_resp = log_ll - log_sum[:, np.newaxis]
        return log_resp, float(log_sum.mean())

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray):
        n = X.shape[0]
        resp = np.exp(log_resp)
        Nk = resp.sum(axis=0) + 1e-300

        self.weights_ = Nk / n
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.variances_[k] = (resp[:, k] @ (diff ** 2)) / Nk[k]

    def fit(self, X: np.ndarray) -> "GaussianMixture":
        rng = np.random.RandomState(self.random_state)
        self._init_kmeans_pp(X, rng)

        prev_ll = -np.inf
        for i in range(1, self.max_iter + 1):
            log_resp, mean_ll = self._e_step(X)
            self._m_step(X, log_resp)
            if abs(mean_ll - prev_ll) < self.tol:
                self.converged_ = True
                self.n_iter_ = i
                break
            prev_ll = mean_ll

        if not self.converged_:
            self.n_iter_ = self.max_iter
        return self

    def score(self, X: np.ndarray) -> float:
        # среднее log-likelihood по выборке (ПМП)
        n, K = X.shape[0], self.n_components
        log_ll = np.empty((n, K))
        for k in range(K):
            log_ll[:, k] = np.log(self.weights_[k] + 1e-300) + self._log_gauss(X, k)
        return float(np.logaddexp.reduce(log_ll, axis=1).mean())

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_resp, _ = self._e_step(X)
        return np.argmax(log_resp, axis=1)

    def bic(self, X: np.ndarray) -> float:
        n, d = X.shape
        K = self.n_components
        n_params = (K - 1) + 2 * K * d
        return -2.0 * self.score(X) * n + n_params * np.log(n)

    def aic(self, X: np.ndarray) -> float:
        n, d = X.shape
        K = self.n_components
        n_params = (K - 1) + 2 * K * d
        return -2.0 * self.score(X) * n + 2.0 * n_params
