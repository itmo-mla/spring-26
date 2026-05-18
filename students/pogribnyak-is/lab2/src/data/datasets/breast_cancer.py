import numpy as np
from sklearn.datasets import load_breast_cancer
from pandas import DataFrame

from data.dataset import Dataset


class BreastCancerDataset(Dataset):
    def load(self):
        data = load_breast_cancer()
        self.df = DataFrame(data.data, columns=data.feature_names)
        self.df["target"] = data.target

    def preprocess(self):
        self.df["target"] = np.where(self.df["target"] == 0, -1, 1)
        self.df.astype(np.float32)
