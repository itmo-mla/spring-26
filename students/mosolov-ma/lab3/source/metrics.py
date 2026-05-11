from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class MetricsSummary:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


@dataclass
class ModelBenchmark:
    name: str
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_metrics: MetricsSummary
    train_time_seconds: float


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> MetricsSummary:
    roc_auc = float("nan")
    if y_proba is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    return MetricsSummary(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
    )


def stratified_cv_score(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for train_idx, val_idx in splitter.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        scores.append(float(accuracy_score(y[val_idx], y_pred)))
    return float(np.mean(scores)), float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0)


def pretty_metrics(summary: MetricsSummary) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(summary).items()}
