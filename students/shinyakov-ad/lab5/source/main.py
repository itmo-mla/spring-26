from pathlib import Path
import os
import time

LAB_DIR = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = LAB_DIR / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
from data_load import load_dataset
from metrics import ndcg_at_k, rmse
from model import KarypisSLIMReference, LSM, SLIM, SklearnLatentSemanticModel

RANDOM_STATE = 42

def save_metric_plot(summary, metric, output_path, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(summary["model"], summary[metric])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def evaluate_model(name, model, dataset):
    start_time = time.time()

    model.fit(dataset)

    runtime = time.time() - start_time
    
    predictions = model.predict_pairs(dataset["test_rows"], dataset["test_cols"])

    return {
        "model": name,
        "rmse": rmse(dataset["test_ratings"], predictions),
        "ndcg_10": ndcg_at_k(
            dataset["test_rows"],
            dataset["test_ratings"],
            predictions,
            k=10,
        ),
        "runtime_sec": runtime,
    }


def main():
    artifacts_dir = LAB_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(random_state=RANDOM_STATE)

    models = [
        ("custom_slim", SLIM(random_state=RANDOM_STATE)),
        ("karypis_slim_reference", KarypisSLIMReference()),
        ("custom_lsm", LSM(random_state=RANDOM_STATE)),
        ("sklearn_lsm", SklearnLatentSemanticModel(random_state=RANDOM_STATE)),
    ]

    rows = [evaluate_model(name, model, dataset) for name, model in models]
    summary = pd.DataFrame(rows)

    save_metric_plot(summary, "rmse", artifacts_dir / "rmse.png", "RMSE comparison", "RMSE")
    save_metric_plot(summary, "ndcg_10", artifacts_dir / "ndcg_10.png", "NDCG@10 comparison", "NDCG@10")
    save_metric_plot(summary, "runtime_sec", artifacts_dir / "runtime.png", "Runtime comparison", "seconds")

    print("Recommendation models comparison:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
