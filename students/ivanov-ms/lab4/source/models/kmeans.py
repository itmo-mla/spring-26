import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

        self.centroids = None
        self.labels_ = None

    def initialize_centroids(self, X):
        n_samples = X.shape[0]
        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)

        self.centroids = X[random_idx]

    def compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            diff = X - self.centroids[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)

        return distances

    def assign_clusters(self, X):
        distances = self.compute_distances(X)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]

            if len(cluster_points) == 0:
                new_centroids[k] = self.centroids[k]
            else:
                new_centroids[k] = cluster_points.mean(axis=0)

        return new_centroids

    def fit(self, X):
        self.initialize_centroids(X)

        for _ in range(self.max_iter):
            labels = self.assign_clusters(X)

            new_centroids = self.update_centroids(X, labels)
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break
        return self

    def predict(self, X):
        return self.assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
