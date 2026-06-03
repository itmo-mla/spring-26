import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.data_loader import load_ratings, train_test_split, build_sparse_matrix
from source.slim import SLIM
from source.lfm import LFM
from source.baselines import SurpriseSVD, KarypisSLIM
from source.metrics import compute_rmse_from_df, compute_ndcg
from source.visualization import (
    plot_rating_distribution,
    plot_user_activity,
    plot_rmse_comparison,
    plot_ndcg_comparison,
    plot_lfm_training_curve,
    plot_slim_weight_sparsity,
)

SLIM_ALPHA = 0.1
SLIM_L1_RATIO = 0.5

LFM_N_FACTORS = 50
LFM_N_EPOCHS = 20
LFM_LR = 0.005
LFM_REG = 0.02

NDCG_K = 10
NDCG_MAX_USERS = 300

SAMPLE_USERS = 500    # ограничение датасета для разумного времени обучения


def main():
    print("\n[1/6] Загрузка и предобработка данных")
    df, n_users, n_items = load_ratings(
        min_ratings_per_user=20,
        min_ratings_per_item=300,
        sample_users=SAMPLE_USERS,
    )
    plot_rating_distribution(df)
    plot_user_activity(df)

    train_df, test_df = train_test_split(df, test_fraction=0.2, random_state=42)

    # Разреженная матрица (нужна для SLIM и NDCG)
    train_R = build_sparse_matrix(train_df, n_users, n_items)

    print("\n[2/6] Обучение SLIM (собственная реализация)")
    slim = SLIM(alpha=SLIM_ALPHA, l1_ratio=SLIM_L1_RATIO, max_iter=1000)
    slim.fit(train_R)
    plot_slim_weight_sparsity(slim)

    slim_rmse = compute_rmse_from_df(lambda df: slim.predict_df(df, train_R), test_df)
    print(f"SLIM  RMSE: {slim_rmse:.4f}")

    def slim_recommend_fn(user_idx, top_n):
        return slim.recommend(train_R, user_idx, top_n=top_n, exclude_seen=True)

    slim_ndcg = compute_ndcg(
        slim_recommend_fn, train_df, test_df,
        n_users, k=NDCG_K, max_users=NDCG_MAX_USERS,
    )
    print(f"SLIM  NDCG@{NDCG_K}: {slim_ndcg:.4f}")


    print("\n[3/6] Обучение LFM/SGD (собственная реализация)")
    lfm = LFM(
        n_factors=LFM_N_FACTORS,
        n_epochs=LFM_N_EPOCHS,
        lr=LFM_LR,
        reg_p=LFM_REG,
        reg_q=LFM_REG,
        reg_b=LFM_REG,
        verbose=1,
    )
    lfm.fit(train_df, n_users, n_items)
    plot_lfm_training_curve(lfm)

    lfm_rmse = compute_rmse_from_df(lfm.predict, test_df)
    print(f"LFM   RMSE: {lfm_rmse:.4f}")

    def lfm_recommend_fn(user_idx, top_n):
        return lfm.recommend(user_idx, train_R, top_n=top_n, exclude_seen=True)

    lfm_ndcg = compute_ndcg(
        lfm_recommend_fn, train_df, test_df,
        n_users, k=NDCG_K, max_users=NDCG_MAX_USERS,
    )
    print(f"LFM   NDCG@{NDCG_K}: {lfm_ndcg:.4f}")

    print("\n[4/6] Обучение эталонных моделей (библиотека surprise)")

    ref_svd = SurpriseSVD(
        n_factors=LFM_N_FACTORS,
        n_epochs=LFM_N_EPOCHS,
        lr_all=LFM_LR,
        reg_all=LFM_REG,
    )
    ref_svd.fit(train_df)
    ref_svd_rmse = compute_rmse_from_df(ref_svd.predict, test_df)
    print(f"Surprise SVD RMSE: {ref_svd_rmse:.4f}")

    karpy_slim_rmse = None
    karpy_slim_ndcg = None
    try:
        ref_kslim = KarypisSLIM(l1r=1.0, l2r=1.0)
        ref_kslim.fit(train_R)
        karpy_slim_rmse = compute_rmse_from_df(
            lambda df: ref_kslim.predict(df, train_R), test_df
        )
        print(f"KarypisLab/SLIM RMSE: {karpy_slim_rmse:.4f}")

        def kslim_recommend_fn(user_idx, top_n):
            return ref_kslim.recommend(user_idx, train_R, top_n=top_n)

        karpy_slim_ndcg = compute_ndcg(
            kslim_recommend_fn, train_df, test_df,
            n_users, k=NDCG_K, max_users=NDCG_MAX_USERS,
        )
        print(f"KarypisLab/SLIM NDCG@{NDCG_K}: {karpy_slim_ndcg:.4f}")

    except ImportError as e:
        print(f"KarypisLab/SLIM не установлен")
        print(f"    {e}")

    all_items = list(range(n_items))

    def ref_svd_recommend_fn(user_idx, top_n):
        seen = set(train_df[train_df["user_idx"] == user_idx]["item_idx"].tolist())
        return ref_svd.recommend(user_idx, all_items, seen, top_n=top_n)

    ref_svd_ndcg = compute_ndcg(
        ref_svd_recommend_fn, train_df, test_df,
        n_users, k=NDCG_K, max_users=NDCG_MAX_USERS,
    )
    print(f"Surprise SVD NDCG@{NDCG_K}: {ref_svd_ndcg:.4f}")

    print("\n[5/6] Итоговые результаты")

    rmse_results = {
        "SLIM\n(наш)": slim_rmse,
        "LFM SGD\n(наш)": lfm_rmse,
        "SVD\n(surprise)": ref_svd_rmse,
    }
    if karpy_slim_rmse is not None:
        rmse_results["KarypisSLIM\n(эталон)"] = karpy_slim_rmse

    ndcg_results = {
        f"SLIM\n(наш)": slim_ndcg,
        f"LFM SGD\n(наш)": lfm_ndcg,
        f"SVD\n(surprise)": ref_svd_ndcg,
    }
    if karpy_slim_ndcg is not None:
        ndcg_results["KarypisSLIM\n(эталон)"] = karpy_slim_ndcg

    print("\n  RMSE (ниже = лучше):")
    for name, val in rmse_results.items():
        print(f"    {name.replace(chr(10), ' '):25s}: {val:.4f}")

    print(f"\n  NDCG@{NDCG_K} (выше = лучше):")
    for name, val in ndcg_results.items():
        print(f"    {name.replace(chr(10), ' '):25s}: {val:.4f}")

    print("\n[6/6] Сохранение графиков")
    plot_rmse_comparison(rmse_results)
    plot_ndcg_comparison(ndcg_results, k=NDCG_K)


if __name__ == "__main__":
    main()