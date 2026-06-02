import numpy as np
from tqdm import tqdm

def ndcg_at_k(relevances, k=10):
    relevances = np.array(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    dcg = np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    ideal = np.sort(relevances)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, ideal.size + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def compute_ndcg_for_model(predictions_matrix, test_df, test_users, n_items, desc="NDCG"):
    ndcg_list = []
    for u in tqdm(test_users, desc=desc):
        user_items = test_df[test_df['user_id'] == u]
        if len(user_items) < 2:
            continue
        if isinstance(predictions_matrix, list):
            scores_dict = predictions_matrix[u]
            all_preds = np.array([scores_dict.get(i, 0.0) for i in range(n_items)])
        else:
            all_preds = predictions_matrix[u]
        top_indices = np.argsort(all_preds)[::-1][:10]
        true_ratings = []
        for i in top_indices:
            rating = test_df[(test_df['user_id'] == u) & (test_df['item_id'] == i)]['rating']
            true_ratings.append(rating.values[0] if len(rating) > 0 else 0)
        ndcg_list.append(ndcg_at_k(true_ratings, k=10))
    return np.mean(ndcg_list)