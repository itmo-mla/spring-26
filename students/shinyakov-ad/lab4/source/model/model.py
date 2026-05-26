import numpy as np
from .gaussian import log_multivariate_gaussian
from .init_param import random_init


class GMM:
    def __init__(
        self,
        n_components=3,
        max_iter=100,
        tol=1e-4,
        reg_covar=1e-6,
        random_state=42,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_history_ = []
        self.responsibilities_history_ = []

    def _e_step(self, X):
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_resp[:, k] = (
                np.log(self.weights_[k] + 1e-12)
                + log_multivariate_gaussian(
                    X,
                    self.means_[k],
                    self.covariances_[k],
                    self.reg_covar,
                )
            )

        max_log = np.max(log_resp, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(
            np.sum(np.exp(log_resp - max_log), axis=1, keepdims=True)
        )

        log_likelihood = np.sum(log_sum_exp)
        responsibilities = np.exp(log_resp - log_sum_exp)

        return responsibilities, log_likelihood

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        nk = responsibilities.sum(axis=0) + 1e-12

        self.weights_ = nk / n_samples
        self.means_ = np.einsum("ki,kj->ij", responsibilities, X) / nk[:, np.newaxis]

        covariances = []
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k][:, np.newaxis] * diff
            cov = np.einsum("ki,kj->ij", weighted_diff, diff) / nk[k]
            cov += self.reg_covar * np.eye(n_features)
            covariances.append(cov)

        self.covariances_ = np.array(covariances)

    def _check_convergence(self, old_responsibilities, new_responsibilities):
        if old_responsibilities is None:
            return False

        diff = np.abs(new_responsibilities - old_responsibilities)
        return np.max(diff) < self.tol

    def fit(self, X):
        self.weights_, self.means_, self.covariances_ = random_init(
            X,
            self.n_components,
            self.random_state,
        )

        prev_responsibilities = None
        self.log_likelihood_history_ = []
        self.responsibilities_history_ = []

        for _ in range(self.max_iter):
            responsibilities, log_likelihood = self._e_step(X)
            self._m_step(X, responsibilities)

            self.log_likelihood_history_.append(log_likelihood)
            self.responsibilities_history_.append(responsibilities.copy())

            if self._check_convergence(prev_responsibilities, responsibilities):
                break

            prev_responsibilities = responsibilities

        return self

    def predict_proba(self, X):
        responsibilities, _ = self._e_step(X)
        return responsibilities

    def predict(self, X):
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)

    def score_samples(self, X):
        log_resp = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):
            log_resp[:, k] = (
                np.log(self.weights_[k] + 1e-12)
                + log_multivariate_gaussian(
                    X,
                    self.means_[k],
                    self.covariances_[k],
                    self.reg_covar,
                )
            )

        max_log = np.max(log_resp, axis=1, keepdims=True)
        return (
            max_log
            + np.log(np.sum(np.exp(log_resp - max_log), axis=1, keepdims=True))
        ).ravel()
