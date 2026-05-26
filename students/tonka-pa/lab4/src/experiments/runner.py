from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from sklearn.decomposition import PCA

from src.gmm import MyGaussianMixture, MyGMMClassifier
from src.metrics import (
    classification_metrics,
    density_metrics,
    rows_to_dataframe,
    rows_to_markdown,
)
from src.preprocess import DataSplits, make_splits
from src.visualization import (
    plot_bar_comparison,
    plot_confusion_matrix,
    plot_density_2d,
    plot_density_contour_2d,
    plot_log_likelihood_curve,
    plot_metric_vs_k,
    plot_roc,
)

from .config import ExperimentConfig
from .model_selection import select_best_k
from .sklearn_baselines import (
    GaussianNB,
    QuadraticDiscriminantAnalysis,
    SklearnGMMClassifier,
    make_density_baseline,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "experiments"

console = Console()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_metrics(rows: list[dict], out_dir: Path, name: str = "metrics") -> None:
    df = rows_to_dataframe(rows)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    md = rows_to_markdown(rows)
    (out_dir / f"{name}.md").write_text(md + "\n")


def _save_params(config: ExperimentConfig, out_dir: Path) -> None:
    payload = {
        "name": config.name,
        "task": config.task,
        "data": config.data,
        "model": config.model,
        "baseline": config.baseline,
        "runs": config.runs,
        "artifacts": config.artifacts,
    }
    (out_dir / "params.json").write_text(json.dumps(payload, indent=2, default=str))


def _make_data(config: ExperimentConfig) -> DataSplits:
    data_kwargs = dict(config.data)
    data_kwargs.setdefault("n_samples", 50_000)
    data_kwargs.setdefault("test_size", 0.2)
    data_kwargs.setdefault("val_size", 0.1)
    data_kwargs.setdefault("scale", True)
    data_kwargs.setdefault("stratify", True)
    data_kwargs.setdefault("random_state", 0)
    data_kwargs.setdefault("pca_components", None)
    return make_splits(**data_kwargs)


# ----------------------------------------------------------------- density
def _fit_and_time(model, X) -> tuple[float, float]:
    t0 = time.perf_counter()
    model.fit(X)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.score(X)
    score_time = time.perf_counter() - t0
    return fit_time, score_time


def _run_density(config: ExperimentConfig, splits: DataSplits, out_dir: Path) -> None:
    model_cfg = dict(config.model)
    baseline_cfg = dict(config.baseline) if config.baseline else dict(model_cfg)
    baseline_cfg.pop("verbose", None)
    random_state = int(model_cfg.get("random_state", 0))

    custom = MyGaussianMixture(**model_cfg)
    custom_fit_time, custom_score_time = _fit_and_time(custom, splits.X_train)
    custom_metrics = density_metrics(custom, splits.X_train, splits.X_val, splits.X_test)

    sk = make_density_baseline(
        n_components=int(model_cfg.get("n_components", 1)),
        covariance_type=model_cfg.get("covariance_type", "full"),
        random_state=random_state,
        n_init=int(model_cfg.get("n_init", 1)),
        init_params=model_cfg.get("init_params", "kmeans"),
        max_iter=int(model_cfg.get("max_iter", 100)),
        tol=float(model_cfg.get("tol", 1e-3)),
        reg_covar=float(model_cfg.get("reg_covar", 1e-6)),
    )
    sk_fit_time, sk_score_time = _fit_and_time(sk, splits.X_train)
    sk_metrics = {
        "log_lik_train": float(sk.score(splits.X_train)),
        "log_lik_val": float(sk.score(splits.X_val)),
        "log_lik_test": float(sk.score(splits.X_test)),
        "bic_train": float(sk.bic(splits.X_train)),
        "aic_train": float(sk.aic(splits.X_train)),
    }

    rows = [
        {
            "model": "MyGaussianMixture",
            "covariance_type": custom.covariance_type,
            "K": custom.n_components,
            **custom_metrics,
            "fit_time": custom_fit_time,
            "score_time": custom_score_time,
            "n_iter": custom.n_iter_,
            "converged": custom.converged_,
        },
        {
            "model": "sklearn.GaussianMixture",
            "covariance_type": sk.covariance_type,
            "K": sk.n_components,
            **sk_metrics,
            "fit_time": sk_fit_time,
            "score_time": sk_score_time,
            "n_iter": int(sk.n_iter_),
            "converged": bool(sk.converged_),
        },
    ]
    _save_metrics(rows, out_dir)

    # convergence figure (custom only — sklearn does not expose history)
    if hasattr(custom, "loss_history_"):
        plot_log_likelihood_curve(
            custom.loss_history_,
            out_dir / "figures" / "custom" / "log_likelihood_convergence.png",
            title=f"EM convergence (K={custom.n_components}, {custom.covariance_type})",
        )

    plot_bar_comparison(
        {"custom (train)": custom_metrics["log_lik_train"], "sklearn (train)": sk_metrics["log_lik_train"]},
        out_dir / "figures" / "log_likelihood_train.png",
        title="train per-sample log-likelihood",
        ylabel="log-likelihood",
    )
    plot_bar_comparison(
        {"custom (val)": custom_metrics["log_lik_val"], "sklearn (val)": sk_metrics["log_lik_val"]},
        out_dir / "figures" / "log_likelihood_val.png",
        title="val per-sample log-likelihood",
        ylabel="log-likelihood",
    )

    # 2D visualization: project to 2D for scatter (separate from any PCA in the pipeline)
    _density_2d_plots(splits.X_train, custom, out_dir)


def _density_2d_plots(X_train: np.ndarray, model: MyGaussianMixture, out_dir: Path) -> None:
    """Produce a 2D visualization of the fitted clusters via an auxiliary PCA.

    If the data is already 2D, no extra PCA is needed.
    """
    if X_train.shape[1] == 2:
        X2 = X_train
    else:
        proj = PCA(n_components=2, random_state=0).fit(X_train)
        X2 = proj.transform(X_train)
    # Cluster assignments on the original-feature model
    labels = model.predict(X_train)
    plot_density_2d(
        X2,
        labels,
        out_dir / "figures" / "pca2d_clusters.png",
        title="EM cluster assignment (PCA-2D scatter)",
    )

    # For an honest density contour we fit a separate 2D GMM with the same K/covariance.
    aux = MyGaussianMixture(
        n_components=model.n_components,
        covariance_type=model.covariance_type if model.covariance_type != "spherical" else "full",
        random_state=0,
        n_init=1,
        init_params="kmeans",
    )
    aux.fit(X2)
    plot_density_contour_2d(
        aux,
        X2,
        out_dir / "figures" / "pca2d_density_contour.png",
        title="GMM density contour in PCA-2D (auxiliary 2D fit)",
    )


# -------------------------------------------------------- model selection
def _run_model_selection(config: ExperimentConfig, splits: DataSplits, out_dir: Path) -> None:
    k_grid = list(config.model.get("k_grid", [1, 2, 3, 5, 7, 10, 15, 20]))
    covariance_type = config.model.get("covariance_type", "diag")
    common = dict(config.model)
    common.pop("k_grid", None)
    custom_rows, sklearn_rows = select_best_k(
        splits=splits,
        k_grid=k_grid,
        covariance_type=covariance_type,
        common=common,
    )
    rows = [*custom_rows, *sklearn_rows]
    _save_metrics(rows, out_dir, name="metrics")

    # plots: bic, aic, val log-likelihood, fit time vs K
    for metric, ylabel in [
        ("bic", "BIC (lower = better)"),
        ("aic", "AIC (lower = better)"),
        ("log_lik_val", "val log-likelihood (higher = better)"),
        ("fit_time", "fit time (s)"),
    ]:
        plot_metric_vs_k(
            k_grid,
            {
                "custom": [r[metric] for r in custom_rows],
                "sklearn": [r[metric] for r in sklearn_rows],
            },
            out_dir / "figures" / f"{metric}_vs_k.png",
            title=f"{metric} vs K ({covariance_type})",
            ylabel=ylabel,
        )


# -------------------------------------------------------- classification
def _classifier_run(name: str, model, splits: DataSplits) -> dict:
    t0 = time.perf_counter()
    model.fit(splits.X_train, splits.y_train)
    fit_time = time.perf_counter() - t0
    y_pred = model.predict(splits.X_test)
    y_proba = model.predict_proba(splits.X_test)
    pos_index = list(model.classes_).index(1) if 1 in model.classes_ else -1
    y_score = y_proba[:, pos_index]
    metrics = classification_metrics(splits.y_test, y_pred, y_score=y_score)
    return {"model": name, **metrics, "fit_time": fit_time, "y_pred": y_pred, "y_score": y_score}


def _run_classifier(config: ExperimentConfig, splits: DataSplits, out_dir: Path) -> None:
    rows: list[dict] = []
    figures_dir = out_dir / "figures"

    for spec in config.runs:
        K = int(spec["n_components"])
        ct = spec["covariance_type"]
        seed = int(spec.get("random_state", 0))
        n_init = int(spec.get("n_init", 1))

        custom_name = f"MyGMMClassifier (K={K}, {ct})"
        custom = MyGMMClassifier(
            n_components=K,
            covariance_type=ct,
            random_state=seed,
            n_init=n_init,
            init_params=spec.get("init_params", "kmeans"),
            max_iter=int(spec.get("max_iter", 100)),
            tol=float(spec.get("tol", 1e-3)),
            reg_covar=float(spec.get("reg_covar", 1e-6)),
        )
        custom_result = _classifier_run(custom_name, custom, splits)
        rows.append({k: v for k, v in custom_result.items() if k not in ("y_pred", "y_score")})

        plot_confusion_matrix(
            splits.y_test,
            custom_result["y_pred"],
            figures_dir / "custom" / f"confusion_K{K}_{ct}.png",
            title=custom_name,
        )
        plot_roc(
            splits.y_test,
            custom_result["y_score"],
            figures_dir / "custom" / f"roc_K{K}_{ct}.png",
            title=custom_name,
        )

        # paired baseline
        baseline_kind = spec.get("baseline", "gmm")
        if baseline_kind == "gnb":
            bl = GaussianNB()
            bl_name = "GaussianNB"
        elif baseline_kind == "qda":
            bl = QuadraticDiscriminantAnalysis()
            bl_name = "QDA"
        else:
            bl = SklearnGMMClassifier(
                n_components=K,
                covariance_type=ct,
                random_state=seed,
                n_init=n_init,
            )
            bl_name = f"sklearn.GMMClassifier (K={K}, {ct})"
        bl_result = _classifier_run(bl_name, bl, splits)
        rows.append({k: v for k, v in bl_result.items() if k not in ("y_pred", "y_score")})

        plot_confusion_matrix(
            splits.y_test,
            bl_result["y_pred"],
            figures_dir / "sklearn" / f"confusion_K{K}_{ct}_{baseline_kind}.png",
            title=bl_name,
        )
        plot_roc(
            splits.y_test,
            bl_result["y_score"],
            figures_dir / "sklearn" / f"roc_K{K}_{ct}_{baseline_kind}.png",
            title=bl_name,
        )

    _save_metrics(rows, out_dir)

    # one bar chart comparing accuracy of all runs
    acc_dict = {row["model"]: row["accuracy"] for row in rows}
    plot_bar_comparison(
        acc_dict,
        figures_dir / "accuracy_comparison.png",
        title="accuracy on test split",
        ylabel="accuracy",
    )
    rocauc_dict = {row["model"]: row.get("roc_auc", float("nan")) for row in rows}
    plot_bar_comparison(
        rocauc_dict,
        figures_dir / "rocauc_comparison.png",
        title="ROC-AUC on test split",
        ylabel="ROC-AUC",
    )


# ------------------------------------------------------------------ main
def run_experiment(config_path: Path) -> Path:
    from .config import load_config

    config = load_config(config_path)
    out_dir = _ensure_dir(RESULTS_DIR / config.name)
    _ensure_dir(out_dir / "figures")
    _ensure_dir(out_dir / "tables")
    _save_params(config, out_dir)

    console.rule(f"[bold]{config.name}[/bold] ({config.task})")
    splits = _make_data(config)
    console.print(
        f"data: train={splits.X_train.shape} val={splits.X_val.shape} test={splits.X_test.shape}"
    )

    if config.task == "density":
        _run_density(config, splits, out_dir)
    elif config.task == "model_selection":
        _run_model_selection(config, splits, out_dir)
    elif config.task == "classification":
        _run_classifier(config, splits, out_dir)
    else:
        raise ValueError(f"unknown task: {config.task}")

    console.print(f"[green]saved[/green] -> {out_dir}")
    return out_dir
