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

    feature_names = X.columns.tolist()

    # добавляем пропуски
    # mask = np.random.rand(*X.shape) < 0.05
    # X = X.mask(mask)
    X = X.to_numpy(dtype=float)
    y = y.to_numpy()

    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution (full dataset):")
    for cls, cnt in zip(unique, counts):
        print(f"Class {cls}: {cnt}")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("\nTrain distribution:")
    for cls, cnt in zip(unique_train, counts_train):
        print(f"Class {cls}: {cnt}")

    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("\nTest distribution:")
    for cls, cnt in zip(unique_test, counts_test):
        print(f"Class {cls}: {cnt}")

    return X_train, X_test, y_train, y_test, feature_names