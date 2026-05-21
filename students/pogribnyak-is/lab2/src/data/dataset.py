from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame

from data.scaler import DefaultScaler


class Dataset(ABC):
    def __init__(self, target_col: str = 'target', seed: int = 42):
        self.seed = seed
        self.target_col = target_col
        self.scaler: DefaultScaler = DefaultScaler()

        self.df: DataFrame = DataFrame()
        self.X: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.y: np.ndarray = np.empty((0,), dtype=np.float32)
        self.X_train: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.X_val: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.X_test: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.y_train: np.ndarray = np.empty((0,), dtype=np.float32)
        self.y_val: np.ndarray = np.empty((0,), dtype=np.float32)
        self.y_test: np.ndarray = np.empty((0,), dtype=np.float32)

        self.load_data()

    @abstractmethod
    def load(self) -> DataFrame: pass

    @abstractmethod
    def preprocess(self) -> DataFrame: pass

    def load_data(self):
        self.load()
        self.preprocess()
        self.X = self.df.drop(columns=[self.target_col]).to_numpy(dtype=np.float32)
        self.y = self.df[self.target_col].to_numpy()

    def split_indices(self, n_samples, test_size=0.2, val_size=0.25):
        np.random.seed(self.seed)
        idx = np.random.permutation(n_samples)

        test_split = int(n_samples * test_size)
        val_split = int(n_samples * val_size)

        test_idx = idx[:test_split]
        val_idx = idx[test_split:test_split + val_split]
        train_idx = idx[test_split + val_split:]

        return train_idx, val_idx, test_idx

    def scale_data(self):
        scaler = DefaultScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

    def split_and_scale(self, test_size=0.2, val_size=0.25):
        train_idx, val_idx, test_idx = self.split_indices(len(self.X), test_size, val_size)

        self.X_train, self.y_train = self.X[train_idx], self.y[train_idx]
        self.X_val, self.y_val = self.X[val_idx], self.y[val_idx]
        self.X_test, self.y_test = self.X[test_idx], self.y[test_idx]

        self.scale_data()

        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

