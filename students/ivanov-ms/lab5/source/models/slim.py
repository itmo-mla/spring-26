from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet

from .base import BaseRanker, PredictResult
from utils.utils import save_to_mmap


def _train_item(i, R_path, R_shape, R_dtype, l1_coef, l2_coef, train_kwargs):
    """Train ElasticNet for one item"""
    # Open memmap in "read-only" mode (without coping to memory)
    R = np.memmap(R_path, dtype=R_dtype, mode='r', shape=R_shape)

    X, y = np.delete(R, i, axis=1), R[:, i]

    if train_kwargs.get("use_mask", False):
        mask = y > 0
        if train_kwargs.get("sample_perc", 0) > 0:
            sample_perc = train_kwargs.pop("sample_perc")
            cnt_mask = mask.sum()
            sample_idx = np.random.choice(y.shape[0], int(cnt_mask * sample_perc), replace=False)
            mask[sample_idx] = True

        X, y = X[mask], y[mask]
        if y.shape[0] <= 2:
            print(f"Not enough data for {i}: {y.shape[0]}")
            return i, np.zeros(X.shape[1] + 1)

    if "use_mask" in train_kwargs:
        train_kwargs.pop("use_mask")
    if "sample_perc" in train_kwargs:
        train_kwargs.pop("sample_perc")

    model = ElasticNet(
        alpha=l1_coef + l2_coef,
        l1_ratio=l1_coef / (l1_coef + l2_coef),
        fit_intercept=False,
        **train_kwargs
    )
    model.fit(X, y)

    return i, np.insert(model.coef_, i, 0)


class SLIM(BaseRanker):
    def __init__(self, l1_coef: float = 0.1, l2_coef: float = 0.1, n_processes: int = 1, **train_kwargs):
        super(SLIM, self).__init__()
        self.n_processes = n_processes
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.train_kwargs = train_kwargs
        self.W = None

    def fit(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]):
        if isinstance(R, pd.DataFrame):
            self.items = np.array(list(R.columns))
            R = R.to_numpy()

        n_items = R.shape[1]
        self.W = np.zeros((n_items, n_items))

        # Save R matrix to memmap
        with save_to_mmap(R, name="R") as R_params:
            # Run parallel training using joblib
            results = Parallel(
                n_jobs=self.n_processes, backend='loky', verbose=0
            )(
                delayed(_train_item)(
                    i, R_params["R_path"], R_params["R_shape"], R_params["R_dtype"],
                    self.l1_coef, self.l2_coef, self.train_kwargs
                )
                for i in tqdm(range(n_items))
            )

            for i, coefs in results:
                self.W[i, :] = coefs

        return self

    def predict(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]):
        if self.W is None:
            raise RuntimeError("Model not trained yet")

        if isinstance(R, pd.DataFrame):
            users_ids = np.array(R.index)
            R = R.to_numpy()
        else:
            users_ids = np.arange(R.shape[0])

        result = PredictResult(users_ids, self.items, R @ self.W)
        return result

    def __str__(self):
        return f"SLIM[" \
                   f"fitted={self.W is not None}," \
                   f"l1_coef={self.l1_coef}," \
                   f"l2_coef={self.l2_coef}," \
                   f"train_kwargs={self.train_kwargs}" \
               f"]"
