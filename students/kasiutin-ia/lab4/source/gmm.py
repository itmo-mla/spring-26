import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal


class GMM:

    def __init__(self, n_components: int, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter

    def initialize(self, X: np.array, init_type: str):
        self.component_weights = np.full(self.n_components, 1 / self.n_components)  
        self.g = np.zeros((len(X), self.n_components))

        if init_type == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_components, random_state=0).fit(X)
            self.means = kmeans.cluster_centers_
            self.cov = np.array([np.cov(X[kmeans.labels_ == i].T) for i in range(self.n_components)])

        elif init_type == 'random':
            self.means = X[np.random.choice(len(X), self.n_components, replace=False)]
            global_cov = np.cov(X.T)
            self.cov = np.array([global_cov.copy() for _ in range(self.n_components)])
        

    def e_step(self, X: np.array):
        for i in range(len(X)):
            for j in range(self.n_components):
                self.g[i, j] = (
                    self.component_weights[j] * multivariate_normal.pdf(X[i], self.means[j], self.cov[j]) 
                    / np.sum([self.component_weights[j] * multivariate_normal.pdf(X[i], self.means[j], self.cov[j]) for j in range(self.n_components)])
                )

    def m_step(self, X: np.array):
        for j in range(self.n_components):
            self.means[j] = 1 / (len(X) * self.component_weights[j]) * np.sum(self.g[:, j][:, np.newaxis] * X, axis=0)

            diff = X - self.means[j]
            self.cov[j] = 1 / (len(X) * self.component_weights[j]) * (self.g[:, j][:, np.newaxis] * diff).T @ diff
            self.cov[j] += np.eye(X.shape[1]) * 1e-6

        self.component_weights = 1 / len(X) * self.g.sum(axis=0)

    def compute_log_likelihood(self, X: np.array):
        log_likelihood = 0

        for i in range(len(X)):
            sample_prob = 0
            for j in range(self.n_components):
                sample_prob += (
                    self.component_weights[j]
                    * multivariate_normal.pdf(
                        X[i],
                        self.means[j],
                        self.cov[j]
                    )
                )

            log_likelihood += np.log(sample_prob + 1e-10)

        return log_likelihood


    def fit(self, X, init_type='kmeans', tol=1e-4):
        self.initialize(X, init_type)
        prev_log_likelihood = -np.inf
        for iteration in range(self.max_iter):

            self.e_step(X)
            self.m_step(X)
            log_likelihood = self.compute_log_likelihood(X)

            print(
                f"Iteration {iteration + 1}, "
                f"log-likelihood: {log_likelihood:.6f}"
            )

            if abs(log_likelihood - prev_log_likelihood) < tol:
                print("Finished")
                break

            prev_log_likelihood = log_likelihood

        return self


    def predict(self, X):
        probabilities = np.zeros((len(X), self.n_components))

        for j in range(self.n_components):
            probabilities[:, j] = (
                self.component_weights[j]
                * multivariate_normal.pdf(
                    X,
                    mean=self.means[j],
                    cov=self.cov[j]
                )
            )

        return np.argmax(probabilities, axis=1)