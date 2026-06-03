import logging
import os
import shutil
import time

from model import SLIM, SLIMRef, SVD, SVDRef
from utils import load_data, compute_rmse, ndcg_at_k, save_metrics, plot_comparison


def main():
    SEED = 42
    K_NDCG = 10
    N_FACTORS = 20

    results_path = os.path.join(os.getcwd(), "students", "chebykin-aa", "lab5", "results")
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(results_path, "main.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Загрузка MovieLens 100K...")
    R_train, R_test, mask_train, mask_test = load_data(random_state=SEED)
    n_users, n_items = R_train.shape
    logging.info(
        f"Датасет: {n_users} пользователей, {n_items} фильмов, "
        f"train={int((R_train > 0).sum())}, test={int(mask_test.sum())}"
    )

    metrics = {}

    logging.info("SLIM (EASE): обучение...")
    t0 = time.perf_counter()
    slim = SLIM(reg=200.0)
    slim.fit(R_train)
    slim_time = time.perf_counter() - t0
    slim_pred = slim.predict_matrix(R_train)
    metrics["SLIM"] = save_metrics(
        os.path.join(results_path, "custom_slim.txt"), "SLIM (EASE)",
        compute_rmse(R_test, slim_pred, mask_test),
        ndcg_at_k(R_test, slim_pred, mask_train, k=K_NDCG),
        slim_time,
    )

    logging.info("SLIMRef (ElasticNet): обучение...")
    t0 = time.perf_counter()
    slim_ref = SLIMRef(alpha=0.1, l1_ratio=0.5, max_iter=200)
    slim_ref.fit(R_train)
    slim_ref_time = time.perf_counter() - t0
    slim_ref_pred = slim_ref.predict_matrix(R_train)
    metrics["SLIMRef"] = save_metrics(
        os.path.join(results_path, "ref_slim.txt"), "SLIMRef (ElasticNet)",
        compute_rmse(R_test, slim_ref_pred, mask_test),
        ndcg_at_k(R_test, slim_ref_pred, mask_train, k=K_NDCG),
        slim_ref_time,
    )

    logging.info("SVD (Funk): обучение...")
    t0 = time.perf_counter()
    svd = SVD(n_factors=N_FACTORS, n_epochs=20, lr=0.005, reg=0.02, random_state=SEED)
    svd.fit(R_train)
    svd_time = time.perf_counter() - t0
    svd_pred = svd.predict_matrix()
    metrics["SVD"] = save_metrics(
        os.path.join(results_path, "custom_svd.txt"), "SVD (Funk SGD)",
        compute_rmse(R_test, svd_pred, mask_test),
        ndcg_at_k(R_test, svd_pred, mask_train, k=K_NDCG),
        svd_time,
    )

    logging.info("SVDRef (TruncatedSVD): обучение...")
    t0 = time.perf_counter()
    svd_ref = SVDRef(n_factors=N_FACTORS, random_state=SEED)
    svd_ref.fit(R_train)
    svd_ref_time = time.perf_counter() - t0
    svd_ref_pred = svd_ref.predict_matrix()
    metrics["SVDRef"] = save_metrics(
        os.path.join(results_path, "ref_svd.txt"), "SVDRef (TruncatedSVD)",
        compute_rmse(R_test, svd_ref_pred, mask_test),
        ndcg_at_k(R_test, svd_ref_pred, mask_train, k=K_NDCG),
        svd_ref_time,
    )

    with open(os.path.join(results_path, "comparison.txt"), "w", encoding="utf-8") as f:
        f.write(f"{'Модель':<12} {'RMSE':>8} {'NDCG@10':>10} {'Время, с':>10}\n")
        for name, m in metrics.items():
            f.write(f"{name:<12} {m['rmse']:>8.4f} {m['ndcg']:>10.4f} {m['time']:>10.2f}\n")

    plot_comparison(metrics, os.path.join(results_path, "comparison.png"))
    print("Pipeline завершён. Результаты сохранены в", results_path)


if __name__ == "__main__":
    main()
