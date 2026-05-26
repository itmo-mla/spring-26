from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

ModelFactory = Callable[[dict[str, Any]], Any]


def run_oob_grid_search(
    model_factory: ModelFactory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict[str, list[Any]],
    base_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select model parameters by fitted model oob_score_."""
    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    for params in ParameterGrid(param_grid):
        trial_params = dict(base_params or {})
        trial_params.update(params)
        trial_params["bootstrap"] = True
        trial_params["oob_score"] = True

        model = model_factory(trial_params)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - started

        oob_score = float(getattr(model, "oob_score_", np.nan))
        row = dict(trial_params)
        row["oob_score"] = oob_score
        row["fit_time_sec"] = fit_time
        rows.append(row)

        if np.isfinite(oob_score) and oob_score > best_score:
            best_score = oob_score
            best_params = dict(trial_params)

    if best_params is None:
        raise RuntimeError("OOB grid search did not produce a valid score.")

    results = pd.DataFrame(rows).sort_values("oob_score", ascending=False)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    return best_params, results.reset_index(drop=True)
