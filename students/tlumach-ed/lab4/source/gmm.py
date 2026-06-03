import numpy as np


class GMM:

    def __init__(self, n_components=3, max_iter=200, tol=1e-4,
                 random_state=None, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.reg_covar = reg_covar

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_history_ = []


    def _init_params(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        k = self.n_components

        self.weights_ = np.full(k, 1.0 / k)

        indices = rng.choice(n_samples, size=k, replace=False)
        self.means_ = X[indices].copy()

        self.covariances_ = np.array([np.eye(n_features) for _ in range(k)])


    def _gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]

        cov_reg = cov + self.reg_covar * np.eye(n_features)

        cov_inv = np.linalg.inv(cov_reg)
        _, log_det = np.linalg.slogdet(cov_reg)

        diff = X - mean

        exponent = -0.5 * np.einsum('ni,ij,nj->n', diff, cov_inv, diff)

        log_norm = -0.5 * (n_features * np.log(2 * np.pi) + log_det)

        return np.exp(log_norm + exponent)


    def _e_step(self, X):
        n_samples = X.shape[0]
        k = self.n_components

        g = np.zeros((n_samples, k))
        for j in range(k):
            g[:, j] = self.weights_[j] * self._gaussian_pdf(
                X, self.means_[j], self.covariances_[j]
            )

        total = g.sum(axis=1, keepdims=True)

        total = np.where(total == 0, 1e-300, total)

        g = g / total
        return g


    def _m_step(self, X, g):
        n_samples, n_features = X.shape
        k = self.n_components

        l_wj = g.sum(axis=0)

        self.weights_ = l_wj / n_samples

        self.means_ = (g.T @ X) / l_wj[:, np.newaxis]

        for j in range(k):
            diff = X - self.means_[j]
            weighted_diff = g[:, j:j+1] * diff
            self.covariances_[j] = (weighted_diff.T @ diff) / l_wj[j]

    def _log_likelihood(self, X):
        p_x = np.zeros(X.shape[0])
        for j in range(self.n_components):
            p_x += self.weights_[j] * self._gaussian_pdf(
                X, self.means_[j], self.covariances_[j]
            )
        p_x = np.where(p_x <= 0, 1e-300, p_x)
        return np.sum(np.log(p_x))

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._init_params(X)
        self.log_likelihood_history_ = []

        prev_ll = -np.inf

        for _ in range(self.max_iter):
            g = self._e_step(X)

            self._m_step(X, g)

            ll = self._log_likelihood(X)
            self.log_likelihood_history_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self


    def score(self, X):
        X = np.asarray(X, dtype=float)
        return self._log_likelihood(X) / X.shape[0]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        g = self._e_step(X)
        return np.argmax(g, axis=1)