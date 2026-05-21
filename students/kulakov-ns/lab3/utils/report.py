import json
from typing import Dict, Any



def make_report(my_result: Dict[str, Any], sklearn_result: Dict[str, Any], my_time: float, sklearn_time: float) -> None:
    with open("data/report_template.md", "r", encoding="utf-8") as file:
        content = file.read()

    content = content.format(
        my_params=json.dumps(my_result["best_params_"], ensure_ascii=False),
        sk_params=json.dumps(sklearn_result["best_params_"], ensure_ascii=False),
        my_cv=my_result["best_score_"],
        sk_cv=sklearn_result["best_score_"],
        my_acc=my_result["metrics"]["accuracy"],
        my_prec=my_result["metrics"]["precision"],
        my_rec=my_result["metrics"]["recall"],
        my_f1=my_result["metrics"]["f1"],
        my_logloss=my_result["metrics"]["logloss"],
        my_time=my_time,
        sk_acc=sklearn_result["metrics"]["accuracy"],
        sk_prec=sklearn_result["metrics"]["precision"],
        sk_rec=sklearn_result["metrics"]["recall"],
        sk_f1=sklearn_result["metrics"]["f1"],
        sk_logloss=sklearn_result["metrics"]["logloss"],
        sk_time=sklearn_time,
    )

    with open("data/report.md", "w", encoding="utf-8") as file:
        file.write(content)



def save_results(my_result: Dict[str, Any], sklearn_result: Dict[str, Any], my_time: float, sklearn_time: float) -> None:
    with open("data/results.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "my_best_params": my_result["best_params_"],
                "my_best_cv_score": my_result["best_score_"],
                "my_metrics": my_result["metrics"],
                "my_train_time": my_time,
                "sklearn_best_params": sklearn_result["best_params_"],
                "sklearn_best_cv_score": sklearn_result["best_score_"],
                "sklearn_metrics": sklearn_result["metrics"],
                "sklearn_train_time": sklearn_time,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
