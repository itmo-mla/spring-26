import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


class GaussianMixture:
    def __init__(self, n_components: int = 3, max_iter: int = 200,
                 tol: float = 1e-6, random_state: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "GaussianMixture":
        self._init_params(X)
        self.log_likelihoods_: list[float] = []
        prev = -np.inf
        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self.score(X)
            self.log_likelihoods_.append(ll)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        return self

    def score(self, X: np.ndarray) -> float:
        return logsumexp(self._log_prob(X), axis=1).mean()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._e_step(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._e_step(X).argmax(axis=1)

    def _init_params(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        chosen = [int(rng.integers(n))]
        for _ in range(self.n_components - 1):
            dists = np.min([np.sum((X - X[i]) ** 2, axis=1) for i in chosen], axis=0)
            chosen.append(int(rng.choice(n, p=dists / dists.sum())))
        self.means_: np.ndarray = X[chosen].astype(float)
        self.covariances_: np.ndarray = np.stack([np.cov(X.T)] * self.n_components)
        self.weights_: np.ndarray = np.full(self.n_components, 1.0 / self.n_components)

    def _log_prob(self, X: np.ndarray) -> np.ndarray:
        log_p = np.empty((len(X), self.n_components))
        for k in range(self.n_components):
            log_p[:, k] = np.log(self.weights_[k] + 1e-300) + multivariate_normal.logpdf(
                X, self.means_[k], self.covariances_[k], allow_singular=True
            )
        return log_p

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        log_p = self._log_prob(X)
        return np.exp(log_p - logsumexp(log_p, axis=1, keepdims=True))

    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        n, d = X.shape
        nk = resp.sum(axis=0).clip(1e-10)
        self.weights_ = nk / n
        self.means_ = (resp.T @ X) / nk[:, None]
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (resp[:, k:k+1] * diff).T @ diff / nk[k]
            self.covariances_[k] += 1e-6 * np.eye(d)
