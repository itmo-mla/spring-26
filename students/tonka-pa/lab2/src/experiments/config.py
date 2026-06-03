from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path("configs")
DEFAULT_CONFIGS = [
    CONFIG_DIR / "rf_default.yaml",
    CONFIG_DIR / "rf_shallow.yaml",
    CONFIG_DIR / "rf_many_trees.yaml",
    CONFIG_DIR / "rf_oob_grid_search.yaml",
    CONFIG_DIR / "rf_best.yaml",
]

DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "max_samples": None,
    "oob_score": True,
    "class_weight": None,
    "n_jobs": None,
}

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200],
    "max_features": ["sqrt", 0.5],
    "max_depth": [None, 12],
    "min_samples_leaf": [1, 3],
    "criterion": ["gini"],
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one experiment config."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping.")

    config = {
        "experiment_name": loaded.get("experiment_name", config_path.stem),
        "task": loaded.get("task", "classification"),
        "random_state": loaded.get("random_state", 42),
        "test_size": loaded.get("test_size", 0.2),
        "run_grid_search": loaded.get("run_grid_search", False),
        "use_grid_search_best_params": loaded.get(
            "use_grid_search_best_params",
            False,
        ),
        "compute_oob_importance": loaded.get("compute_oob_importance", True),
        "include_decision_tree": loaded.get("include_decision_tree", True),
        "config_path": str(config_path),
    }

    model_params = dict(DEFAULT_MODEL_PARAMS)
    model_params.update(loaded.get("model_params") or {})
    config["model_params"] = model_params
    config["param_grid"] = loaded.get("param_grid") or DEFAULT_PARAM_GRID

    _validate_config(config)
    return config


def list_default_configs() -> list[Path]:
    """Return config files used by --run-all in a stable order."""
    return DEFAULT_CONFIGS


def _validate_config(config: dict[str, Any]) -> None:
    if config["task"] != "classification":
        raise ValueError("Only classification is supported in this lab.")
    if not config["experiment_name"]:
        raise ValueError("experiment_name must be non-empty.")
    if not 0 < float(config["test_size"]) < 1:
        raise ValueError("test_size must be in (0, 1).")
    if config["model_params"].get("oob_score") and not config["model_params"].get(
        "bootstrap",
        True,
    ):
        raise ValueError("OOB score requires bootstrap=True.")
    if config["run_grid_search"] and not config["param_grid"]:
        raise ValueError("Grid-search config must define param_grid.")
