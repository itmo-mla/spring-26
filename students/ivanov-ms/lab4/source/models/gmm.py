import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from .kmeans import KMeans


def log_likelihood(X, means, covariances, weights):
    likelihood = 0
    for k in range(means.shape[0]):
        pdf = multivariate_normal.pdf(
            X, mean=means[k], cov=covariances[k], allow_singular=True
        )
        likelihood += weights[k] * pdf
    return np.sum(np.log(likelihood))


class GaussianMixtureModel:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, init_params: str = 'kmeans'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_params = init_params

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.train_ll_hist_ = []

    def initialize(self, X):
        n_samples, n_features = X.shape
        if self.init_params == "random":
            random_idx = np.random.choice(n_samples, self.n_components, replace=False)
            self.means_ = X[random_idx]
            self.covariances_ = np.array([
                np.cov(X.T) for _ in range(self.n_components)
            ])
            self.weights_ = np.ones(self.n_components) / self.n_components
        elif self.init_params == "kmeans":
            kmeans = KMeans(n_clusters=self.n_components, max_iter=self.max_iter, tol=self.tol)
            labels = kmeans.fit_predict(X)

            self.means_ = kmeans.centroids.copy()
            self.weights_ = np.zeros(self.n_components)
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))

            for k in range(self.n_components):
                cluster_points = X[labels == k]
                Nk = len(cluster_points)
                self.weights_[k] = Nk / n_samples
                if Nk > 1:
                    cov = np.cov(cluster_points.T)
                else:
                    cov = np.eye(n_features)
                cov += 1e-6 * np.eye(n_features)
                self.covariances_[k] = cov
        else:
            raise ValueError(f"Incorrect init type: {self.init_params}, possible values are: kmeans, random")

    def e_step(self, X):
        n_samples = X.shape[0]

        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            pdf = multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covariances_[k], allow_singular=True
            )
            responsibilities[:, k] = self.weights_[k] * pdf

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        n_samples, n_features = X.shape

        Nk = responsibilities.sum(axis=0)
        self.weights_ = Nk / n_samples
        self.means_ = (
            responsibilities.T @ X
        ) / Nk[:, np.newaxis]
        self.covariances_ = []

        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = (
                responsibilities[:, k][:, np.newaxis] * diff
            ).T @ diff / Nk[k]
            # Regularization
            cov += 1e-6 * np.eye(n_features)
            self.covariances_.append(cov)

        self.covariances_ = np.array(self.covariances_)

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.initialize(X)
        self.train_ll_hist_ = []

        for i in range(self.max_iter):
            responsibilities = self.e_step(X)
            self.m_step(X, responsibilities)
            ll = log_likelihood(X, self.means_, self.covariances_, self.weights_)

            if len(self.train_ll_hist_) > 0:
                if abs(ll - self.train_ll_hist_[-1]) < self.tol:
                    self.train_ll_hist_.append(ll)
                    break

            self.train_ll_hist_.append(ll)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)

    def score(self, X):
        return log_likelihood(X, self.means_, self.covariances_, self.weights_)

    def __str__(self):
        is_fitted = self.means_ is not None and self.covariances_ is not None and self.weights_ is not None
        return (
            f"{self.__class__.__name__}["
            f"fitted={is_fitted}; n_components={self.n_components}; "
            f"max_iter={self.max_iter}; tol={self.tol}"
            f"]"
        )
