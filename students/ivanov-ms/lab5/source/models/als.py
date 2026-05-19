from typing import List, Tuple, Union

import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseRanker, PredictResult
from utils.utils import Columns, save_to_mmap

warnings.filterwarnings('ignore')


def _solve_batch_factors(
    batch_idx: List[int],
    X_path: str, X_shape, X_dtype,
    select_path: str, select_shape, select_dtype,
    lambda_reg: float, X_cols: Tuple[int, int, int]
):
    # Read arrays in read-only mode
    X = np.memmap(X_path, dtype=X_dtype, mode='r', shape=X_shape)
    select_factor = np.memmap(select_path, dtype=select_dtype, mode='r', shape=select_shape)

    index_col, select_col, rating_col = X_cols
    n_factors = select_factor.shape[1]
    reg_term = lambda_reg * np.eye(n_factors)

    new_factors = np.zeros((len(batch_idx), n_factors), dtype=select_dtype)

    for i, idx in enumerate(batch_idx):
        # select other factors & ratings
        select_mask = X[index_col] == idx
        factors_selected = select_factor[X[select_col, select_mask].astype(int)]
        ratings = X[rating_col, select_mask]
        # Get inv term and projection
        inv_term = factors_selected.T @ factors_selected + reg_term
        project_term = factors_selected.T @ ratings
        # solve linear equation & update target factors
        new_factors[i] = np.linalg.solve(inv_term, project_term)

    return batch_idx, new_factors


class ALS(BaseRanker):
    def __init__(self, n_factors: int = 20, n_epochs: int = 50, lambda_reg: float = 0.1,
                 n_processes: int = 1, batch_size: int = 32):
        super(ALS, self).__init__()

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lambda_reg = lambda_reg
        self.n_processes = n_processes
        self.batch_size = batch_size

        self.users_map = {}
        self.users_factors = None
        self.items_factors = None

        self.train_history = {}

    def _init_factors(self, df, matrix: bool = False):
        if matrix:
            self.users_map = {u: i for i, u in enumerate(df.index)}
            self.items_map = {it: i for i, it in enumerate(df.columns)}
        else:
            self.users_map = {u: i for i, u in enumerate(df[Columns.User].unique())}
            self.items_map = {it: i for i, it in enumerate(df[Columns.Item].unique())}

        self.users_factors = np.random.rand(len(self.users_map), self.n_factors) / 100
        self.items_factors = np.random.rand(len(self.items_map), self.n_factors) / 100

    def _solve_factor(
            self, target_factor: np.ndarray, select_factor: np.ndarray,
            cols: Tuple[int, int, int], X_params: dict
    ):
        n_targets = target_factor.shape[0]
        n_batches = int(np.ceil(n_targets / self.batch_size))
        batches = [
            [i for i in range(b_i * self.batch_size, min((b_i + 1) * self.batch_size, n_targets))]
            for b_i in range(n_batches)
        ]

        with save_to_mmap(select_factor, "select") as select_params:
            args = [X_params["X_path"], X_params["X_shape"], X_params["X_dtype"]]
            args += [select_params["select_path"], select_params["select_shape"], select_params["select_dtype"]]
            args += [self.lambda_reg, cols]
            results = Parallel(n_jobs=self.n_processes, backend='loky', verbose=0)(
                delayed(_solve_batch_factors)(batch_idx, *args)
                for batch_idx in batches
            )

        for batch_idx, coefs in results:
            target_factor[batch_idx, :] = coefs

        return target_factor

    def fit(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]):
        if isinstance(R, pd.DataFrame):
            self.items = np.array(list(R.columns))
            self._init_factors(R, matrix=True)
            R = R.to_numpy()
        else:
            self._init_factors(df, matrix=False)

        self.train_history = {"epoch": [], "loss": [], "loss_reg": []}

        X = np.array([
            df[Columns.User].map(self.users_map).to_numpy(),
            df[Columns.Item].map(self.items_map).to_numpy(),
            df[Columns.Rating].to_numpy()
        ])
        user_col, item_col, rating_col = 0, 1, 2

        user_factors_cols = (user_col, item_col, rating_col)
        item_factor_cols = (item_col, user_col, rating_col)

        with save_to_mmap(X, name="X") as X_params:
            for epoch in range(self.n_epochs):
                # solve user factors
                self.users_factors = self._solve_factor(
                    target_factor=self.users_factors, select_factor=self.items_factors,
                    cols=user_factors_cols, X_params=X_params
                )
                # solve items factors
                self.items_factors = self._solve_factor(
                    target_factor=self.items_factors, select_factor=self.users_factors,
                    cols=item_factor_cols, X_params=X_params
                )

                preds = self.users_factors @ self.items_factors.T
                loss = np.mean((R - preds)**2)

                loss_reg = loss + self.lambda_reg * ((self.users_factors ** 2).sum() + (self.items_factors ** 2).sum())

                self.train_history["epoch"].append(epoch + 1)
                self.train_history["loss"].append(loss)
                self.train_history["loss_reg"].append(loss_reg)

                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}, Loss with reg.: {loss_reg:.4f}")

        return self

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

    def __str__(self):
        return f"ALS[" \
                   f"fitted={self.users_factors is not None}," \
                   f"n_factors={self.n_factors}," \
                   f"lambda_coef={self.lambda_reg}" \
               f"]"

