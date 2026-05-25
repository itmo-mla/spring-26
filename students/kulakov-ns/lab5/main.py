import time
from typing import Any, Dict

from models.my_lsa import get_my_lsa
from models.my_slim import get_my_slim
from models.sklearn_lsa import get_sklearn_lsa
from models.sklearn_slim import get_sklearn_slim
from utils.dataset import load_text_recommendation_dataset
from utils.metrics import evaluate_model
from utils.report import make_report, save_results


def train_model(train_function, dataset):
    start_time = time.perf_counter()
    grid_search = train_function(dataset)
    train_time = time.perf_counter() - start_time
    return grid_search, train_time


def build_result(model_name: str, grid_search, dataset: Dict[str, Any]):
    metrics = evaluate_model(
        model_name,
        grid_search.best_estimator_,
        dataset["train_matrix"],
        dataset["test_interactions"],
        k=10,
    )
    return {
        "best_params_": grid_search.best_params_,
        "best_score_": grid_search.best_score_,
        "metrics": metrics,
    }


def main():
    dataset = load_text_recommendation_dataset()

    experiments = {
        "my_slim": ("My SLIM", get_my_slim),
        "sklearn_slim": ("Sklearn ElasticNet SLIM", get_sklearn_slim),
        "my_lsa": ("My LSA/SVD", get_my_lsa),
        "sklearn_lsa": ("Sklearn TruncatedSVD", get_sklearn_lsa),
    }

    results = {}
    train_times = {}

    for key, (model_name, train_function) in experiments.items():
        grid_search, train_time = train_model(train_function, dataset)
        results[key] = build_result(model_name, grid_search, dataset)
        train_times[key] = train_time

    dataset_info = {
        "n_users": dataset["n_users"],
        "n_items": dataset["n_items"],
        "n_interactions": len(dataset["interactions"]),
        "n_train": len(dataset["train_interactions"]),
        "n_test": len(dataset["test_interactions"]),
        "topics": ", ".join(sorted(dataset["items"]["topic"].unique().tolist())),
    }

    make_report(results, train_times, dataset_info)
    save_results(results, train_times, dataset_info)


if __name__ == "__main__":
    main()
