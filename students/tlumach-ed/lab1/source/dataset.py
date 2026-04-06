import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_dataset():
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame
    y = (df["class"] == ">50K").astype(int)
    X = df.drop(columns=["class"])

    # категориальные в one-hot
    X = pd.get_dummies(X)

    # добавляем пропуски
    mask = np.random.rand(*X.shape) < 0.05
    X = X.mask(mask)
    X = X.to_numpy(dtype=float)
    y = y.to_numpy()

    return train_test_split(X, y, test_size=0.3, random_state=42)