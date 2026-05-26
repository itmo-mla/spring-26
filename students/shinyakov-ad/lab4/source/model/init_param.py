import numpy as np
from sklearn.cluster import KMeans


def random_init(X, n_components, random_state=42):
    n_samples, n_features = X.shape

    kmeans = KMeans(
        n_clusters=n_components,
        random_state=random_state,
        n_init=10,
    )
    kmeans.fit(X)

    weights = np.ones(n_components) / n_components
    means = kmeans.cluster_centers_
    covariances = np.array([np.eye(n_features) for _ in range(n_components)])

    return weights, means, covariances
