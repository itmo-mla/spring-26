from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.mixture import GaussianMixture

from data import load_wine_density_dataset, make_projection
from gmm import GaussianMixtureEM
from metrics import (
    evaluate_gmm,
    plot_component_weights,
    plot_likelihood_comparison,
    plot_log_likelihood,
    plot_metrics,
    plot_projection,
)


LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"


def fit_with_time(model, X):
    started = time.perf_counter()
    model.fit(X)
    return model, time.perf_counter() - started


def markdown_metrics_table(metrics: pd.DataFrame) -> list[str]:
    rows = [
        "| Model | Train avg LL | Test avg LL | Test total LL | BIC | AIC | Clustering acc | ARI | NMI | Iter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        rows.append(
            "| {model} | {train_avg_log_likelihood:.4f} | {test_avg_log_likelihood:.4f} | "
            "{test_total_log_likelihood:.2f} | {bic_test:.2f} | {aic_test:.2f} | "
            "{clustering_accuracy:.4f} | {adjusted_rand_index:.4f} | "
            "{normalized_mutual_info:.4f} | {n_iter} |".format(**row)
        )
    return rows


def make_report(bundle, args, metrics: pd.DataFrame, timings: dict[str, float]) -> str:
    rows = [
        "# Experiment summary",
        "",
        f"Dataset: {bundle.description['name']}",
        f"Samples: {bundle.description['samples']}",
        f"Features: {bundle.description['features']}",
        f"Components: {args.n_components}",
        "",
        "## Parameters",
        "",
        f"`n_components={args.n_components}`, `covariance_type=full`, "
        f"`max_iter={args.max_iter}`, `tol={args.tol}`, `reg_covar={args.reg_covar}`",
        "",
        "## Metrics",
        "",
        *markdown_metrics_table(metrics),
        "",
        "## Training time",
        "",
        f"- Custom GMM EM: `{timings['custom']:.4f}` seconds",
        f"- Sklearn GaussianMixture: `{timings['sklearn']:.4f}` seconds",
        "",
    ]
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 4: EM algorithm for GMM")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--reg-covar", type=float, default=1e-5)
    parser.add_argument("--n-init", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_wine_density_dataset(test_size=args.test_size, random_state=args.random_state)

    custom_gmm, custom_time = fit_with_time(
        GaussianMixtureEM(
            n_components=args.n_components,
            max_iter=args.max_iter,
            tol=args.tol,
            reg_covar=args.reg_covar,
            n_init=args.n_init,
            random_state=args.random_state,
        ),
        bundle.X_train,
    )
    sklearn_gmm, sklearn_time = fit_with_time(
        GaussianMixture(
            n_components=args.n_components,
            covariance_type="full",
            max_iter=args.max_iter,
            tol=args.tol,
            reg_covar=args.reg_covar,
            n_init=args.n_init,
            init_params="kmeans",
            random_state=args.random_state,
        ),
        bundle.X_train,
    )

    metrics = pd.DataFrame(
        [
            evaluate_gmm("Custom GMM EM", custom_gmm, bundle.X_train, bundle.X_test, bundle.y_test),
            evaluate_gmm("Sklearn GaussianMixture", sklearn_gmm, bundle.X_train, bundle.X_test, bundle.y_test),
        ]
    )
    metrics["train_time_sec"] = [custom_time, sklearn_time]
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)

    params = pd.DataFrame(
        {
            "component": list(range(args.n_components)),
            "custom_weight": custom_gmm.weights_,
            "sklearn_weight": sklearn_gmm.weights_,
        }
    )
    params.to_csv(ARTIFACTS_DIR / "component_weights.csv", index=False)

    summary = {
        "dataset": bundle.description,
        "train_size": len(bundle.X_train),
        "test_size": len(bundle.X_test),
        "parameters": vars(args),
        "custom_converged": custom_gmm.converged_,
        "sklearn_converged": sklearn_gmm.converged_,
        "custom_train_time_sec": custom_time,
        "sklearn_train_time_sec": sklearn_time,
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    timings = {"custom": custom_time, "sklearn": sklearn_time}
    report = make_report(bundle, args, metrics, timings)
    (ARTIFACTS_DIR / "experiment_summary.md").write_text(report, encoding="utf-8")

    _, _, test_2d, _ = make_projection(bundle.X_train, bundle.X_test, bundle.X_all)
    plot_log_likelihood(custom_gmm.lower_bounds_, ARTIFACTS_DIR / "em_convergence.png")
    plot_metrics(metrics, ARTIFACTS_DIR / "clustering_metrics.png")
    plot_likelihood_comparison(metrics, ARTIFACTS_DIR / "likelihood_comparison.png")
    plot_projection(
        test_2d,
        bundle.y_test,
        custom_gmm.predict(bundle.X_test),
        sklearn_gmm.predict(bundle.X_test),
        ARTIFACTS_DIR / "pca_clusters.png",
    )
    plot_component_weights(custom_gmm.weights_, sklearn_gmm.weights_, ARTIFACTS_DIR / "component_weights.png")

    print(report)
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
