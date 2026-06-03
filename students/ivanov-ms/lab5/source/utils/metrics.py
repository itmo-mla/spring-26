import numpy as np
import pandas as pd

from .utils import Columns


def mertics_at_k(true_pairs: pd.DataFrame, pred_pairs: pd.DataFrame, k: int):
    true_pairs = true_pairs.sort_values([Columns.User, Columns.Rating], ascending=[True, False])
    true_pairs['true_rank'] = true_pairs.groupby(Columns.User).cumcount() + 1
    true_pairs = true_pairs[true_pairs['true_rank'] <= k]

    merged_df = true_pairs.merge(pred_pairs, on=[Columns.User, Columns.Item], how='outer')

    not_nan_true = ~merged_df[Columns.Rating].isna()
    rmse = np.sqrt(np.mean(
        (merged_df.loc[not_nan_true, Columns.Rating] - merged_df.loc[not_nan_true, Columns.Score].fillna(0))**2)
    )

    min_rating, max_rating = true_pairs[Columns.Rating].min(), true_pairs[Columns.Rating].max()

    def _metrics(group):
        group = group.sort_values(Columns.Rank)
        nn_true = ~group[Columns.Rating].isna()
        nn_pred = ~group[Columns.Score].isna()

        TP = (nn_true & nn_pred).sum()
        precision = TP / k
        recall = TP / len(group[Columns.Rating])

        user_norm_rate = (group[Columns.Rating] - min_rating) / (max_rating - min_rating)
        avg_rating = np.nanmean(user_norm_rate)
        user_rel = (user_norm_rate - avg_rating + 1).fillna(0)

        # print("User rel:", user_rel)
        # print("User rel nn_pred:", user_rel[nn_pred])
        # print("User rel nn_true:", user_rel[nn_true])

        inv_logs = 1 / np.log2(np.arange(1, k + 1) + 1)
        dcg = np.sum((np.exp2(user_rel[nn_pred]) - 1) * inv_logs)
        idcg = np.sum((np.exp2(user_rel[nn_true]) - 1) * inv_logs[:nn_true.sum()])
        # print(f"DCG: {dcg}, idcg: {idcg}\n")

        if idcg == 0:
            ndcg = 1
        else:
            ndcg = dcg / idcg
        return pd.Series({f"Precision@{k}": precision, f"Recall@{k}": recall, f"NDCG@{k}": ndcg})

    metrics = merged_df.groupby(Columns.User).apply(_metrics)
    metrics = metrics.mean().to_dict()
    metrics[f"RMSE@{k}"] = rmse

    return metrics

