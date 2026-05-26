from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def build_sklearn_model(task: str, model_params: dict[str, Any]) -> Any:
    """Return a configured sklearn GradientBoosting model."""
    params = _filter_params(model_params)
    if task == "classification":
        return GradientBoostingClassifier(**params)
    if task == "regression":
        return GradientBoostingRegressor(**params)
    raise ValueError(f"Unsupported task: {task!r}")


def _filter_params(model_params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "subsample",
        "random_state",
    }
    return {k: v for k, v in model_params.items() if k in allowed}
