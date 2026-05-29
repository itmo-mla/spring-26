from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares

from .base import BaseRanker, PredictResult
from utils.utils import Columns


class ReferenceALS(BaseRanker):
    def __init__(self, n_factors: int = 20, n_epochs: int = 50, lambda_reg: float = 0.1, n_processes: int = 1):
        super(ReferenceALS, self).__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lambda_reg = lambda_reg
        self.n_processes = n_processes

        self.model = AlternatingLeastSquares(
            factors=n_factors,
            regularization=lambda_reg,
            iterations=n_epochs,
            calculate_training_loss=True,
            num_threads=n_processes
        )
        self.users_map = None
        self.users_factors = None
        self.items_factors = None

    def fit(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]):
        if isinstance(R, pd.DataFrame):
            self.items = np.array(list(R.columns))
            self.users_map = {u: i for i, u in enumerate(R.index)}
            R = R.to_numpy()
        else:
            self.items = df[Columns.Item].unique()
            self.users_map = {u: i for i, u in enumerate(df[Columns.User].unique())}

        R = csr_matrix(R)
        self.model.fit(R)

        self.users_factors = self.model.user_factors
        self.items_factors = self.model.item_factors

    def predict(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]) -> PredictResult:
        if self.users_factors is None or self.items_factors is None:
            raise RuntimeError("Model not trained yet")

        if isinstance(R, pd.DataFrame):
            users_ids = np.array(R.index)
        else:
            users_ids = df[Columns.User].unique()

        users_idx = [self.users_map[u_id] for u_id in users_ids]
        users_factors = self.users_factors[users_idx]
        preds = users_factors @ self.items_factors.T

        result = PredictResult(users_ids, self.items, preds)
        return result

