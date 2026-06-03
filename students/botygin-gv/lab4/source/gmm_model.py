import numpy as np
from scipy.special import logsumexp
from utils import multivariate_gaussian_log_pdf


class GaussianMixtureEM:

    def __init__(self, n_components=3, max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_ = None
        self.n_iter_ = 0

    def _init_params(self, X):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self.weights_ = np.ones(self.n_components) / self.n_components

        idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])

    def _e_step(self, X):
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_resp[:, k] = np.log(self.weights_[k] + 1e-10) + \
                             multivariate_gaussian_log_pdf(X, self.means_[k], self.covariances_[k])

        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        log_resp -= log_norm
        responsibilities = np.exp(log_resp)

        log_likelihood = np.sum(log_norm)
        return responsibilities, log_likelihood

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0)
        Nk = np.maximum(Nk, 1e-10)

        self.weights_ = Nk / n_samples

        self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]

        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
            self.covariances_[k] += self.reg_covar * np.eye(n_features)

    def fit(self, X):
        self._init_params(X)
        prev_ll = -np.inf

        for i in range(self.max_iter):
            responsibilities, ll = self._e_step(X)
            self._m_step(X, responsibilities)

            self.n_iter_ = i + 1
            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.log_likelihood_ = ll
        return self

    def score(self, X):
        _, ll = self._e_step(X)
        return ll / X.shape[0]

    def predict(self, X):
        resp, _ = self._e_step(X)
        return np.argmax(resp, axis=1)