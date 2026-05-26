import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import lil_matrix
from data_loader import load_movielens_100k
from recommender_models import SLIM, ALS

os.makedirs('images', exist_ok=True)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def ndcg_at_k(y_true, y_pred, k=10):
    order = np.argsort(y_pred)[::-1]
    rel = y_true[order][:k]
    dcg = sum(r / np.log2(i+2) for i, r in enumerate(rel))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = sum(r / np.log2(i+2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0

def main():
    matrix, ratings_df, user_map, item_map, movies = load_movielens_100k()

    train_mat, test_mat = train_test_split(matrix, test_size=0.2, random_state=42)
    test_rows, test_cols = test_mat.nonzero()
    test_vals = test_mat.data

    baseline_preds = np.full_like(test_vals, train_mat.data.mean())
    baseline_rmse = rmse(test_vals, baseline_preds)
    print(f"\nBaseline (global mean) RMSE: {baseline_rmse:.4f}")

    # Нормализация для Our SLIM
    train_mat_lil = train_mat.tolil()
    n_users, n_items = train_mat_lil.shape
    user_means = np.zeros(n_users, dtype=np.float32)
    train_norm_lil = lil_matrix((n_users, n_items), dtype=np.float32)

    for u in range(n_users):
        row = train_mat_lil[u]
        if row.nnz > 0:
            indices = row.rows[0]
            values = row.data[0]
            mu = sum(values) / len(values)
            user_means[u] = mu
            norm_vals = [v - mu for v in values]
            train_norm_lil[u, indices] = norm_vals
    train_mat_norm = train_norm_lil.tocsr()

    print("\n=== Our SLIM (Ridge, alpha=0.5, с нормализацией) ===")
    slim = SLIM(alpha=0.5, positive_only=True, max_iter=300)
    slim.fit(train_mat_norm)

    test_mat_lil = test_mat.tolil()
    test_norm_lil = lil_matrix(test_mat_lil.shape, dtype=np.float32)
    for u in set(test_rows):
        row = test_mat_lil[u]
        if row.nnz > 0:
            indices = row.rows[0]
            values = row.data[0]
            mu = user_means[u]
            norm_vals = [v - mu for v in values]
            test_norm_lil[u, indices] = norm_vals
    test_mat_norm = test_norm_lil.tocsr()

    slim_preds_norm = slim.predict(test_mat_norm)
    slim_test_preds = []
    for u, i in zip(test_rows, test_cols):
        pred = slim_preds_norm[u, i] + user_means[u]
        slim_test_preds.append(np.clip(pred, 1.0, 5.0))
    slim_test_preds = np.array(slim_test_preds)
    slim_rmse = rmse(test_vals, slim_test_preds)
    print(f"Our SLIM RMSE: {slim_rmse:.4f}")

    test_users = np.unique(test_rows)
    slim_ndcg = []
    for u in test_users:
        mask = (test_rows == u)
        items = test_cols[mask]
        y_true = test_mat[u, items].toarray().ravel()
        y_pred_norm = slim_preds_norm[u, items]
        y_pred = y_pred_norm + user_means[u]
        y_pred = np.clip(y_pred, 1.0, 5.0)
        slim_ndcg.append(ndcg_at_k(y_true, y_pred))
    slim_ndcg_mean = np.mean(slim_ndcg)
    print(f"Our SLIM NDCG@10: {slim_ndcg_mean:.4f}")

    # ALS (SGD)
    print("\n=== ALS (SGD, n_factors=30, n_epochs=10, reg=0.15) ===")
    als = ALS(n_factors=30, n_epochs=10, reg=0.15, lr=0.015)
    als.fit(train_mat)
    als_test_preds = als.predict_pair(test_rows, test_cols)
    als_rmse = rmse(test_vals, als_test_preds)
    print(f"ALS RMSE: {als_rmse:.4f}")

    all_preds_als = als.predict_all()
    als_ndcg = []
    for u in test_users:
        mask = (test_rows == u)
        items = test_cols[mask]
        y_true = test_mat[u, items].toarray().ravel()
        y_pred = all_preds_als[u, items]
        als_ndcg.append(ndcg_at_k(y_true, y_pred))
    als_ndcg_mean = np.mean(als_ndcg)
    print(f"ALS NDCG@10: {als_ndcg_mean:.4f}")

    # Эталонный ALS (Surprise)
    print("\n=== Etalon ALS (Surprise library) ===")
    try:
        from surprise import Dataset, Reader, BaselineOnly
        from surprise.model_selection import train_test_split as surprise_split
        from collections import defaultdict

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        trainset, testset = surprise_split(data, test_size=0.2, random_state=42)
        bsl_options = {'method': 'als', 'n_epochs': 15, 'reg_u': 15, 'reg_i': 15}
        etalon_als = BaselineOnly(bsl_options=bsl_options)
        etalon_als.fit(trainset)
        preds = etalon_als.test(testset)

        y_true_etal = [p.r_ui for p in preds]
        y_pred_etal = [p.est for p in preds]
        etalon_als_rmse = rmse(y_true_etal, y_pred_etal)
        print(f"Etalon ALS RMSE: {etalon_als_rmse:.4f}")

        user_true = defaultdict(list)
        user_pred = defaultdict(list)
        for p in preds:
            user_true[p.uid].append(p.r_ui)
            user_pred[p.uid].append(p.est)

        etalon_als_ndcg = []
        for uid in user_true.keys():
            y_true = np.array(user_true[uid])
            y_pred = np.array(user_pred[uid])
            etalon_als_ndcg.append(ndcg_at_k(y_true, y_pred))
        etalon_als_ndcg_mean = np.mean(etalon_als_ndcg)
        print(f"Etalon ALS NDCG@10: {etalon_als_ndcg_mean:.4f}")

    except ImportError:
        print("Surprise not installed. Install with: pip install scikit-surprise")
        etalon_als_rmse = None
        etalon_als_ndcg_mean = None

    # График RMSE
    plt.figure(figsize=(10,6))
    models = ['Baseline', 'Our SLIM', 'Our ALS']
    scores = [baseline_rmse, slim_rmse, als_rmse]
    if etalon_als_rmse is not None:
        models.append('Etalon ALS')
        scores.append(etalon_als_rmse)
    bars = plt.bar(models, scores, color=['gray', 'skyblue', 'orange', 'green'][:len(models)])
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison (lower is better)')
    for bar, val in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center')
    plt.savefig('images/rmse_comparison.png', dpi=150)
    plt.close()

    # График NDCG@10
    plt.figure(figsize=(10,6))
    ndcg_models = ['Our SLIM', 'Our ALS']
    ndcg_scores = [slim_ndcg_mean, als_ndcg_mean]
    if etalon_als_ndcg_mean is not None:
        ndcg_models.append('Etalon ALS')
        ndcg_scores.append(etalon_als_ndcg_mean)
    bars = plt.bar(ndcg_models, ndcg_scores, color=['skyblue', 'orange', 'green'][:len(ndcg_models)])
    plt.ylabel('NDCG@10')
    plt.title('NDCG@10 Comparison (higher is better)')
    for bar, val in zip(bars, ndcg_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center')
    plt.savefig('images/ndcg_comparison.png', dpi=150)
    plt.close()

    # Распределение предсказаний
    plt.figure(figsize=(8,5))
    plt.hist(slim_test_preds, bins=20, alpha=0.7, label='Our SLIM', color='skyblue')
    plt.hist(als_test_preds, bins=20, alpha=0.7, label='Our ALS', color='orange')
    plt.xlabel('Predicted rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of predictions on test set')
    plt.legend()
    plt.savefig('images/predictions_distribution.png', dpi=150)
    plt.close()

    print("\n=== Final Summary ===")
    print(f"Baseline        | RMSE = {baseline_rmse:.4f}")
    print(f"Our SLIM        | RMSE = {slim_rmse:.4f} | NDCG@10 = {slim_ndcg_mean:.4f}")
    print(f"Our ALS         | RMSE = {als_rmse:.4f} | NDCG@10 = {als_ndcg_mean:.4f}")
    if etalon_als_rmse is not None:
        print(f"Etalon ALS      | RMSE = {etalon_als_rmse:.4f} | NDCG@10 = {etalon_als_ndcg_mean:.4f}")
    print("\nГрафики сохранены в папку 'images'.")

if __name__ == "__main__":
    main()