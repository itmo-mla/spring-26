import numpy as np


class GMM:


    def __init__(self, k: int, max_iter: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

        self.weights = None   # w_j,  форма (k,)
        self.means = None     # mu_j, форма (k, n)
        self.covs = None      # Sigma_j, форма (k, n, n)
    
    def _gaussian_pdf(self, X, mean, cov):
        n = X.shape[1]

        diff = X-mean 

        cov += np.eye(n) * 1e-6 

        normalization = np.power(2 * np.pi, n / 2) * np.sqrt(np.linalg.det(cov))

        exponent = -0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)

        return np.exp(exponent) / normalization
    
    def _log_likelihood(self, X):
        likelihood = 0.0
        for j in range(self.k):
            likelihood += self.weights[j] * self._gaussian_pdf(X, self.means[j], self.covs[j])
        return np.sum(np.log(likelihood))

    def _e_step(self, X):
        N = X.shape[0]
        responsibilities = np.zeros((N, self.k))

        for j in range(self.k):
            responsibilities[:, j] = self.weights[j] * self._gaussian_pdf(X, self.means[j], self.covs[j])

        responsibilities_sum = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities /= responsibilities_sum

        return responsibilities


    def _m_step(self, X, responsibilities):
        N_k = np.sum(responsibilities, axis=0)

        self.weights = N_k / X.shape[0]
        self.means = (responsibilities.T @ X) / N_k[:, np.newaxis]
        for j in range(self.k):
            diff = X - self.means[j]
            self.covs[j] = (responsibilities[:, j][:, np.newaxis] * diff).T @ diff / N_k[j]

    def fit(self, X):
        n_features = X.shape[1]
        self.weights = np.ones(self.k) / self.k
        self.means = X[np.random.choice(X.shape[0], self.k, replace=False)]
        self.covs = np.array([np.cov(X, rowvar=False) for _ in range(self.k)])

        log_likelihood_old = None

        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            log_likelihood_new = self._log_likelihood(X)
            if log_likelihood_old is not None and abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new

            
    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
