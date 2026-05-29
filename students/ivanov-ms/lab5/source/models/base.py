from typing import Union

import numpy as np
import pandas as pd

from utils.utils import Columns


class PredictResult:
    def __init__(self, users_ids, partners, scores):
        self.users_ids = users_ids
        self.partners = partners
        self.users_map = dict(zip(list(users_ids), range(len(users_ids))))
        self.partners_map = dict(zip(list(partners), range(len(partners))))
        self.scores = scores

    def get_score(self, user_id, partner) -> float:
        return self.scores[self.users_map[user_id], self.partners_map[partner]]

    def get_scores(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda row: self.get_score(row[Columns.User], row[Columns.Item]), axis=1)


class BaseRanker:
    def __init__(self):
        self.items = None

    def fit(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]):
        raise NotImplementedError()

    def predict(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame]) -> PredictResult:
        raise NotImplementedError()

    def predict_topk(self, df: pd.DataFrame, R: Union[np.ndarray, pd.DataFrame], k: int = 10, explore=0):
        if isinstance(R, pd.DataFrame):
            R = R[self.items]
            users_map = dict(zip(list(R.index), range(R.shape[0])))
            seen_mask = R.to_numpy() > 0
        else:
            users_map = None
            seen_mask = R > 0

        pred_res = self.predict(df, R)
        users_ids, scores = pred_res.users_ids, pred_res.scores

        if explore < k:
            preds = self._prepare_preds(scores, users_ids, k=k)
            if explore <= 0:
                return preds
        else:
            preds = None

        user_idx = np.array([users_map[user_id] for user_id in users_ids]) if users_map is not None else np.arange(len(users_ids))
        scores[seen_mask[user_idx]] = -1e6
        explore_preds = self._prepare_preds(scores, users_ids, k=min(explore, k))
        if preds is None:
            return explore_preds

        preds = self._combine_preds(preds, explore_preds, k, explore)
        return preds, pred_res

    def _prepare_preds(self, scores, users_ids, k=10):
        inds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        scores = np.take_along_axis(scores, inds, axis=1)
        items = self.items[inds] if self.items is not None else inds

        preds = pd.DataFrame({
            Columns.User: np.repeat(users_ids, k),
            Columns.Item: items.flatten(),
            Columns.Score: scores.flatten(),
            Columns.Rank: np.repeat(np.arange(1, k + 1)[None, :], len(users_ids), axis=0).flatten(),
        })
        return preds

    def _combine_preds(self, full_preds, explore_preds, k, explore):
        full_preds = full_preds.rename(columns={Columns.Score: '_full_score'}).drop(Columns.Rank, axis=1)
        explore_preds = explore_preds.rename(columns={Columns.Score: '_explore_score'}).drop(Columns.Rank, axis=1)

        comb_preds = full_preds.merge(explore_preds, on=[Columns.User, Columns.Item], how='outer')

        comb_preds['rn'] = 0
        full_pred = comb_preds['_explore_score'].isna()
        comb_preds.loc[full_pred, 'rn'] = (
            comb_preds[full_pred]
            .sort_values([Columns.User, '_full_score'], ascending=[True, False])
            .groupby(Columns.User).cumcount()
        )

        keep_full = k - explore
        comb_preds = comb_preds.loc[
            comb_preds['rn'] < keep_full, [Columns.User, Columns.Item, '_full_score', '_explore_score']
        ]
        comb_preds[Columns.Score] = comb_preds.pop('_full_score').combine_first(comb_preds.pop('_explore_score'))
        comb_preds = comb_preds.sort_values([Columns.User, Columns.Score], ascending=[True, False])
        comb_preds[Columns.Rank] = comb_preds.groupby(Columns.User).cumcount() + 1

        return comb_preds
