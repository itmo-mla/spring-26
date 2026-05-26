from __future__ import annotations

import time

from src.gmm import MyGaussianMixture
from src.preprocess import DataSplits

from .sklearn_baselines import make_density_baseline


def _row(model_name: str, K: int, covariance_type: str, model, splits: DataSplits, fit_time: float) -> dict:
    log_lik_train = float(model.score(splits.X_train))
    log_lik_val = float(model.score(splits.X_val))
    log_lik_test = float(model.score(splits.X_test))
    return {
        "model": model_name,
        "K": K,
        "covariance_type": covariance_type,
        "log_lik_train": log_lik_train,
        "log_lik_val": log_lik_val,
        "log_lik_test": log_lik_test,
        "bic": float(model.bic(splits.X_train)),
        "aic": float(model.aic(splits.X_train)),
        "fit_time": fit_time,
        "n_iter": int(getattr(model, "n_iter_", -1)),
    }


def select_best_k(
    splits: DataSplits,
    k_grid: list[int],
    covariance_type: str,
    common: dict,
) -> tuple[list[dict], list[dict]]:
    """Sweep ``k_grid`` and fit both the custom GMM and the sklearn baseline."""
    custom_rows: list[dict] = []
    sklearn_rows: list[dict] = []
    for K in k_grid:
        random_state = int(common.get("random_state", 0))
        n_init = int(common.get("n_init", 1))
        max_iter = int(common.get("max_iter", 100))
        tol = float(common.get("tol", 1e-3))
        reg_covar = float(common.get("reg_covar", 1e-6))
        init_params = common.get("init_params", "kmeans")

        custom = MyGaussianMixture(
            n_components=K,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
            init_params=init_params,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
        )
        t0 = time.perf_counter()
        custom.fit(splits.X_train)
        custom_rows.append(_row("custom", K, covariance_type, custom, splits, time.perf_counter() - t0))

        sk = make_density_baseline(
            n_components=K,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
            init_params=init_params,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
        )
        t0 = time.perf_counter()
        sk.fit(splits.X_train)
        sklearn_rows.append(_row("sklearn", K, covariance_type, sk, splits, time.perf_counter() - t0))

    return custom_rows, sklearn_rows
