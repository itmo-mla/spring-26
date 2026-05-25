from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from data import fetch_real_text_corpus, train_test_holdout
from metrics import (
    evaluate_model,
    plot_matrix_density,
    plot_metrics,
    plot_singular_values,
    plot_training_time,
    top_recommendations,
)
from models import NumpyLSARecommender, SklearnLSARecommender, SklearnSlimRecommender, SlimRecommender


LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"


def fit_with_time(model, matrix):
    started = time.perf_counter()
    model.fit(matrix)
    return model, time.perf_counter() - started


def markdown_metrics_table(metrics: pd.DataFrame, ndcg_column: str) -> list[str]:
    rows = [
        "| Model | Family | RMSE | NDCG@10 | Train time, s |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        rows.append(
            f"| {row['model']} | {row['family']} | {row['rmse']:.5f} | "
            f"{row[ndcg_column]:.5f} | {row['train_time_sec']:.4f} |"
        )
    return rows


def make_report(bundle, split, metrics: pd.DataFrame, args: argparse.Namespace) -> str:
    ndcg_column = f"ndcg@{args.ndcg_k}"
    lines = [
        "# Experiment summary",
        "",
        f"Dataset: {bundle.description['name']}",
        f"Documents: {bundle.description['documents']}",
        f"Topics: {bundle.description['topics']}",
        f"Terms: {bundle.description['terms']}",
        f"Full matrix density: {bundle.description['density']:.4f}",
        f"Train matrix density after holdout: {split.density:.4f}",
        f"Held-out entries: {len(split.test_entries)}",
        "",
        "## Parameters",
        "",
        f"`test_fraction={args.test_fraction}`, `lsa_components={args.lsa_components}`, "
        f"`slim_l1={args.slim_l1}`, `slim_l2={args.slim_l2}`",
        "",
        "## Metrics",
        "",
        *markdown_metrics_table(metrics, ndcg_column),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 5: SLIM and latent semantic recommenders")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--docs-per-category", type=int, default=60)
    parser.add_argument("--max-features", type=int, default=180)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--lsa-components", type=int, default=20)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--slim-l1", type=float, default=0.002)
    parser.add_argument("--slim-l2", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = fetch_real_text_corpus(
        docs_per_category=args.docs_per_category,
        max_features=args.max_features,
        random_state=args.random_state,
    )
    split = train_test_holdout(
        bundle.matrix,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )

    models = [
        (
            "Custom SLIM",
            "SLIM",
            SlimRecommender(l1=args.slim_l1, l2=args.slim_l2, max_iter=80),
        ),
        (
            "Sklearn ElasticNet SLIM",
            "SLIM reference",
            SklearnSlimRecommender(alpha=0.001, l1_ratio=0.35),
        ),
        (
            "Custom LSA",
            "Latent semantic",
            NumpyLSARecommender(n_components=args.lsa_components),
        ),
        (
            "Sklearn TruncatedSVD",
            "Latent semantic reference",
            SklearnLSARecommender(n_components=args.lsa_components, random_state=args.random_state),
        ),
    ]

    fitted = []
    rows = []
    for name, family, model in models:
        model, train_time = fit_with_time(model, split.train_matrix)
        prediction = model.predict_all(split.train_matrix)
        rows.append(
            evaluate_model(
                name,
                family,
                prediction,
                split.train_matrix,
                split.test_entries,
                train_time,
                k=args.ndcg_k,
            )
        )
        fitted.append((name, model, prediction))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)
    split.test_entries.to_csv(ARTIFACTS_DIR / "heldout_entries.csv", index=False)

    users = []
    seen_topics = set()
    for idx, topic in enumerate(bundle.topics):
        if topic not in seen_topics:
            users.append(idx)
            seen_topics.add(topic)
    recommendation_frames = [
        top_recommendations(
            prediction,
            split.train_matrix,
            bundle.feature_names,
            bundle.topics,
            users,
            name,
            top_k=8,
        )
        for name, _, prediction in fitted
    ]
    recommendations = pd.concat(recommendation_frames, ignore_index=True)
    recommendations.to_csv(ARTIFACTS_DIR / "sample_recommendations.csv", index=False)

    slim_model = fitted[0][1]
    slim_weights = pd.DataFrame(
        slim_model.coef_,
        index=bundle.feature_names,
        columns=bundle.feature_names,
    )
    top_slim_links = (
        slim_weights.stack()
        .rename("weight")
        .reset_index()
        .rename(columns={"level_0": "source_term", "level_1": "target_term"})
        .query("weight > 0")
        .sort_values("weight", ascending=False)
        .head(40)
    )
    top_slim_links.to_csv(ARTIFACTS_DIR / "top_slim_term_links.csv", index=False)

    summary = {
        "dataset": bundle.description,
        "train_density": split.density,
        "heldout_entries": int(len(split.test_entries)),
        "parameters": vars(args),
    }
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = make_report(bundle, split, metrics, args)
    (ARTIFACTS_DIR / "experiment_summary.md").write_text(report, encoding="utf-8")

    plot_metrics(metrics, ARTIFACTS_DIR / "metrics.png")
    plot_training_time(metrics, ARTIFACTS_DIR / "training_time.png")
    plot_matrix_density(bundle.matrix, split.train_matrix, ARTIFACTS_DIR / "matrix_density.png")
    plot_singular_values(
        {
            "Custom LSA": fitted[2][1].singular_values_,
            "Sklearn TruncatedSVD": fitted[3][1].singular_values_,
        },
        ARTIFACTS_DIR / "singular_values.png",
    )

    print(report)
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
