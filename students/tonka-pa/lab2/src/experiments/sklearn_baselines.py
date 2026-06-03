from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

TREE_PARAMS = {
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "class_weight",
}


def create_sklearn_random_forest(
    params: dict[str, Any],
    random_state: int,
    n_jobs: int | None = None,
) -> RandomForestClassifier:
    """Create the sklearn RandomForestClassifier baseline."""
    rf_params = dict(params)
    rf_params.setdefault("bootstrap", True)
    rf_params.setdefault("oob_score", True)
    rf_params["random_state"] = random_state
    rf_params["n_jobs"] = n_jobs
    return RandomForestClassifier(**rf_params)


def create_decision_tree(
    params: dict[str, Any],
    random_state: int,
) -> DecisionTreeClassifier:
    """Create a single decision tree baseline."""
    tree_params = {key: value for key, value in params.items() if key in TREE_PARAMS}
    tree_params["max_features"] = None
    tree_params["random_state"] = random_state
    return DecisionTreeClassifier(**tree_params)
