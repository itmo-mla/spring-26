import numpy as np
from sklearn.cluster import KMeans


def random_init(X, n_components, random_state=22):
    np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    # K-means для инициализации центров
    kmeans = KMeans(
        n_clusters=n_components,
        random_state=random_state,
        n_init=10
    )
    kmeans.fit(X)
    
    means = kmeans.cluster_centers_
    weights = np.ones(n_components) / n_components
    
    # Единичные ковариационные матрицы (масштабированные)
    covariances = np.array([np.eye(n_features) for _ in range(n_components)])
    
    return weights, means, covariances