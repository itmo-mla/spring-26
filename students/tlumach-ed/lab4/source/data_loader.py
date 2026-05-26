import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def load_wine_dataset():
    data = load_wine()
    X = data.data.astype(float)
    feature_names = list(data.feature_names)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, feature_names


def train_test_split_manual(X, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    return X[idx[n_test:]], X[idx[:n_test]]