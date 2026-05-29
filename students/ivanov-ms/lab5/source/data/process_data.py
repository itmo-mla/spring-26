import numpy as np
import pandas as pd
from typing import Optional, Tuple

from utils.utils import Columns

KEEP_MOVIES = 1000
KEEP_USERS = 100000


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Filter users and movies, short for better performance
    df = df[df['userId'] <= KEEP_USERS]
    df = df[df['movieId'] <= KEEP_MOVIES]
    df = df.rename(columns={
        'userId': Columns.User, 'movieId': Columns.Item,
        'rating': Columns.Rating, 'timestamp': Columns.Datetime
    })
    return df


def train_test_split(df: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    time_vals_idx = np.argsort(df[Columns.Datetime])
    split_lim = round(df.shape[0] * train_size)
    X_train, X_test = df.iloc[time_vals_idx[:split_lim]], df.iloc[time_vals_idx[split_lim:]]
    X_train = X_train.sort_values([Columns.User, Columns.Item])
    X_test = X_test.sort_values([Columns.User, Columns.Item])

    return X_train, X_test


def filter_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Filter users in test only who was in train
    print(f"  Filter train and test, shapes before - train={X_train.shape[0]:,}, test={X_test.shape[0]:,}")
    test_users = X_test[Columns.User].unique()
    X_train = X_train[X_train[Columns.User].isin(test_users)]
    train_users = X_train[Columns.User].unique()
    X_test = X_test[X_test[Columns.User].isin(train_users)]
    # Filter only common items
    train_items = X_train[Columns.Item].unique()
    test_items = X_test[Columns.Item].unique()

    del_train = list(set(train_items) - set(test_items))
    del_test = list(set(test_items) - set(train_items))

    X_train = X_train[~X_train[Columns.Item].isin(del_train)]
    X_test = X_test[~X_test[Columns.Item].isin(del_test)]

    print(f"  Filtered train and test, shapes after - train={X_train.shape[0]:,}, test={X_test.shape[0]:,}")
    return X_train, X_test
