import pandas as pd
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def load_data(train_path="../data/train.csv", test_path=None):
    train_df = pd.read_csv(train_path)

    X = train_df.drop("price_range", axis=1).values
    y = train_df["price_range"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def cross_validate(model_class, X, y, model_params=None, n_splits=5):
    if model_params is None:
        model_params = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []
    times = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)

        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        preds = model.predict(X_val)

        scores.append(accuracy_score(y_val, preds))
        times.append(end - start)

    return {
        "mean_accuracy": np.mean(scores),
        "std_accuracy": np.std(scores),
        "mean_train_time": np.mean(times),
    }
