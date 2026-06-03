from typing import Union

import numpy as np
import pandas as pd
from SLIM import SLIM, SLIMatrix

from .base import BaseRanker, PredictResult
from utils.utils import Columns


class ReferenceSLIM(BaseRanker):
    def __init__(self, l1_coef: float = 0.1, l2_coef: float = 0.1, n_processes: int = 1, **train_kwargs):
        super(ReferenceSLIM, self).__init__()
        self.n_processes = n_processes
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.train_kwargs = train_kwargs

        self.model_params = {
            'dbglvl': 0, 'algo': 'cd',
            'nthreads': self.n_processes,
            'l1r': self.l1_coef, 'l2r': self.l2_coef,
            'optTol': self.train_kwargs.get('tol', 1e-5),
            'niters': train_kwargs.get('max_iter', 500)
        }
        self.model = SLIM()

    def fit(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]):
        if isinstance(R, pd.DataFrame):
            self.items = np.array(list(R.columns))

        trainmat = SLIMatrix(df)
        self.model.train(self.model_params, trainmat)
        return self

    def predict(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]) -> PredictResult:
        users_ids = df[Columns.User].unique()
        trainmat = SLIMatrix(df)
        slim_rec_items, slim_rec_scores = self.model.predict(
            trainmat, nrcmds=len(self.items), returnscores=True
        )
        slim_rec_items = np.array([slim_rec_items[u_id] for u_id in users_ids])
        slim_rec_scores = np.array([slim_rec_scores[u_id] for u_id in users_ids])

        orig_sorted_idx = np.argsort(slim_rec_items, axis=1)
        scores = np.take_along_axis(slim_rec_scores, orig_sorted_idx, axis=1)

        result = PredictResult(users_ids, self.items, scores)
        return result
