import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix
import warnings
warnings.filterwarnings('ignore')

from data import load_movielens100k
from models import train_slim, als_explicit, predict_and_rmse
from etalon import eval_etalon_slim, eval_etalon_als
from metrics import compute_ndcg_for_model
from plotting import (plot_rmse_comparison, plot_als_convergence,
                      plot_als_pred_vs_true, plot_als_residuals,
                      plot_slim_sparsity)

def main():
    ratings_df = load_movielens100k()
    n_users = ratings_df['user_id'].nunique()
    n_items = ratings_df['item_id'].nunique()
    print(f"Пользователей: {n_users}, Товаров: {n_items}, Рейтингов: {len(ratings_df)}")

    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    train_matrix = lil_matrix((n_users, n_items))
    for row in train_df.itertuples():
        train_matrix[row.user_id, row.item_id] = row.rating
    train_matrix = train_matrix.tocsr()

    # Собственный SLIM
    print("\n=== Обучение собственного SLIM ===")
    W_own = train_slim(train_matrix, alpha=0.5, l1_ratio=0.01, max_iter=500, n_jobs=-1)
    y_true_own, y_pred_own = predict_and_rmse(W_own, test_df, train_matrix)
    rmse_own = np.sqrt(np.mean((y_true_own - y_pred_own) ** 2))
    print(f"Собственный SLIM RMSE: {rmse_own:.4f}")

    # Эталонный SLIM
    print("\n=== Эталонный SLIM (KarypisLab) ===")
    try:
        rmse_slim_etalon, user_scores = eval_etalon_slim(train_matrix, test_df, n_items, n_users)
        print(f"Эталонный SLIM RMSE: {rmse_slim_etalon:.4f}")
    except ImportError:
        print("Эталонный SLIM не установлен. Пропускаем.")
        rmse_slim_etalon = None
        user_scores = None

    # Собственный ALS
    print("\n=== Обучение собственного ALS ===")
    U_als, V_als, als_train_rmse = als_explicit(train_matrix, k=50, reg=0.1, iterations=10)

    y_true_als, y_pred_als = [], []
    for row in test_df.itertuples():
        u, i, r = int(row.user_id), int(row.item_id), row.rating
        pred = U_als[u] @ V_als[i]
        y_true_als.append(r)
        y_pred_als.append(pred)
    y_true_als = np.array(y_true_als)
    y_pred_als = np.array(y_pred_als)
    rmse_als_own = np.sqrt(np.mean((y_true_als - y_pred_als) ** 2))
    print(f"Собственный ALS RMSE: {rmse_als_own:.4f}")

    # Эталонный ALS
    print("\n=== Эталонный ALS (implicit) ===")
    try:
        rmse_als_etalon, U_imp, V_imp = eval_etalon_als(train_matrix, test_df, n_users, n_items)
        print(f"Эталонный ALS RMSE: {rmse_als_etalon:.4f}")
    except ImportError:
        print("implicit не установлен. Пропускаем.")
        rmse_als_etalon = None
        U_imp = V_imp = None

    # NDCG@10
    test_users = test_df['user_id'].unique()

    print("\n=== NDCG@10 для собственного SLIM ===")
    pred_matrix_slim_own = train_matrix.toarray().dot(W_own)
    ndcg_slim_own = compute_ndcg_for_model(pred_matrix_slim_own, test_df, test_users, n_items,
                                           desc="NDCG SLIM свой")
    print(f"Средний NDCG@10 (собственный SLIM): {ndcg_slim_own:.4f}")

    if rmse_slim_etalon is not None:
        print("\n=== NDCG@10 для эталонного SLIM ===")
        ndcg_slim_etalon = compute_ndcg_for_model(user_scores, test_df, test_users, n_items,
                                                  desc="NDCG SLIM эталон")
        print(f"Средний NDCG@10 (эталонный SLIM): {ndcg_slim_etalon:.4f}")
    else:
        ndcg_slim_etalon = None

    print("\n=== NDCG@10 для собственного ALS ===")
    pred_matrix_als_own = U_als @ V_als.T
    ndcg_als_own = compute_ndcg_for_model(pred_matrix_als_own, test_df, test_users, n_items,
                                          desc="NDCG ALS свой")
    print(f"Средний NDCG@10 (собственный ALS): {ndcg_als_own:.4f}")

    if rmse_als_etalon is not None:
        print("\n=== NDCG@10 для эталонного ALS ===")
        pred_matrix_als_et = U_imp @ V_imp.T
        ndcg_als_etalon = compute_ndcg_for_model(pred_matrix_als_et, test_df, test_users, n_items,
                                                 desc="NDCG ALS эталон")
        print(f"Средний NDCG@10 (эталонный ALS): {ndcg_als_etalon:.4f}")
    else:
        ndcg_als_etalon = None

    print("\n========== Сводка RMSE ==========")
    print(f"Собственный SLIM RMSE:   {rmse_own:.4f}")
    if rmse_slim_etalon is not None:
        print(f"Эталонный SLIM RMSE:   {rmse_slim_etalon:.4f}")
    print(f"Собственный ALS RMSE:    {rmse_als_own:.4f}")
    if rmse_als_etalon is not None:
        print(f"Эталонный ALS RMSE:     {rmse_als_etalon:.4f}")

    # Графики
    print("\n=== Построение графиков ===")
    plot_rmse_comparison(rmse_own, rmse_slim_etalon, rmse_als_own, rmse_als_etalon)
    plot_als_convergence(als_train_rmse)
    plot_als_pred_vs_true(y_true_als, y_pred_als)
    plot_als_residuals(y_true_als, y_pred_als)
    plot_slim_sparsity(W_own)
    print("Графики сохранены в папке images/")

if __name__ == "__main__":
    main()