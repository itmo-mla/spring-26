import numpy as np
from scipy.stats import multivariate_normal

class CustomGMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar

    def fit(self, X):
        n_samples, n_features = X.shape

        self.weights = np.ones(self.k) / self.k

        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.means = X[random_idx].copy()

        global_cov = np.cov(X, rowvar=False)
        if n_features == 1:
            global_cov = np.array([[global_cov]])
        self.covariances = np.array([global_cov.copy() for _ in range(self.k)])

        self.log_likelihoods = []

        for iteration in range(self.max_iter):
            # E-ШАГ
            resp = np.zeros((n_samples, self.k))
            for j in range(self.k):
                cov_reg = self.covariances[j] + np.eye(n_features) * self.reg_covar
                rv = multivariate_normal(mean=self.means[j], cov=cov_reg, allow_singular=True)
                resp[:, j] = self.weights[j] * rv.pdf(X)
            log_likelihood = np.sum(np.log(np.sum(resp, axis=1) + 1e-10))
            self.log_likelihoods.append(log_likelihood)

            if iteration > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                break

            resp = resp / (np.sum(resp, axis=1, keepdims=True) + 1e-10)

            # M-ШАГ
            N_j = np.sum(resp, axis=0)

            for j in range(self.k):
                self.means[j] = np.sum(resp[:, j:j+1] * X, axis=0) / N_j[j]
                diff = X - self.means[j]
                self.covariances[j] = (resp[:, j:j+1] * diff).T @ diff / N_j[j]
                self.weights[j] = N_j[j] / n_samples

        return self

    def predict_proba(self, X):
        n_samples, n_features = X.shape
        resp = np.zeros((n_samples, self.k))
        for j in range(self.k):
            cov_reg = self.covariances[j] + np.eye(n_features) * self.reg_covar
            rv = multivariate_normal(mean=self.means[j], cov=cov_reg, allow_singular=True)
            resp[:, j] = self.weights[j] * rv.pdf(X)
        resp = resp / (np.sum(resp, axis=1, keepdims=True) + 1e-10)
        return resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X):
        n_samples, n_features = X.shape
        resp = np.zeros((n_samples, self.k))
        for j in range(self.k):
            cov_reg = self.covariances[j] + np.eye(n_features) * self.reg_covar
            rv = multivariate_normal(mean=self.means[j], cov=cov_reg, allow_singular=True)
            resp[:, j] = self.weights[j] * rv.pdf(X)
        return np.mean(np.log(np.sum(resp, axis=1) + 1e-10))


class CustomGaussianNB:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.means = np.zeros((n_classes, n_features))
        self.vars = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[idx, :] = np.mean(X_c, axis=0)
            self.vars[idx, :] = np.var(X_c, axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._classify(x) for x in X]
        return np.array(y_pred)

    def _classify(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            num = np.exp(- (x - self.means[idx])**2 / (2 * self.vars[idx]))
            den = np.sqrt(2 * np.pi * self.vars[idx])
            class_conditional = np.sum(np.log(num / den))

            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
