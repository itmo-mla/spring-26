import json
from typing import Any, Dict


def make_report(results: Dict[str, Dict[str, Any]], train_times: Dict[str, float], dataset_info: Dict[str, Any]) -> None:
    with open("data/report_template.md", "r", encoding="utf-8") as file:
        content = file.read()

    content = content.format(
        n_users=dataset_info["n_users"],
        n_items=dataset_info["n_items"],
        n_interactions=dataset_info["n_interactions"],
        n_train=dataset_info["n_train"],
        n_test=dataset_info["n_test"],
        topics=dataset_info["topics"],
        slim_my_cv=results["my_slim"]["best_score_"],
        slim_my_rmse=results["my_slim"]["metrics"]["rmse"],
        slim_my_ndcg=results["my_slim"]["metrics"]["ndcg@10"],
        slim_my_time=train_times["my_slim"],
        slim_sk_cv=results["sklearn_slim"]["best_score_"],
        slim_sk_rmse=results["sklearn_slim"]["metrics"]["rmse"],
        slim_sk_ndcg=results["sklearn_slim"]["metrics"]["ndcg@10"],
        slim_sk_time=train_times["sklearn_slim"],
        lsa_my_cv=results["my_lsa"]["best_score_"],
        lsa_my_rmse=results["my_lsa"]["metrics"]["rmse"],
        lsa_my_ndcg=results["my_lsa"]["metrics"]["ndcg@10"],
        lsa_my_time=train_times["my_lsa"],
        lsa_sk_cv=results["sklearn_lsa"]["best_score_"],
        lsa_sk_rmse=results["sklearn_lsa"]["metrics"]["rmse"],
        lsa_sk_ndcg=results["sklearn_lsa"]["metrics"]["ndcg@10"],
        lsa_sk_time=train_times["sklearn_lsa"],
        slim_my_params=json.dumps(results["my_slim"]["best_params_"], ensure_ascii=False),
        slim_sk_params=json.dumps(results["sklearn_slim"]["best_params_"], ensure_ascii=False),
        lsa_my_params=json.dumps(results["my_lsa"]["best_params_"], ensure_ascii=False),
        lsa_sk_params=json.dumps(results["sklearn_lsa"]["best_params_"], ensure_ascii=False),
    )

    with open("data/report.md", "w", encoding="utf-8") as file:
        file.write(content)


def save_results(results: Dict[str, Dict[str, Any]], train_times: Dict[str, float], dataset_info: Dict[str, Any]) -> None:
    serializable = {
        "dataset": dataset_info,
        "results": results,
        "train_times": train_times,
    }
    with open("data/results.json", "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)
