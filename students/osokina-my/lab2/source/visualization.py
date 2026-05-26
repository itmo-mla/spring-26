from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

from .random_forest import CustomRandomForestClassifier


def _partial_predict_proba(
    model: CustomRandomForestClassifier,
    X: np.ndarray,
    n_trees: int,
) -> np.ndarray:
    n_samples = X.shape[0]
    n_classes = len(model.classes_)
    proba_sum = np.zeros((n_samples, n_classes))

    for i in range(n_trees):
        tree = model.trees_[i]
        feat_idx = model.feature_indices_per_tree_[i]
        proba_sum += tree.predict_proba(X[:, feat_idx])

    return proba_sum / n_trees


def _collect_loss_history(
    model: CustomRandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> List[float]:
    losses = []
    for k in range(1, len(model.trees_) + 1):
        proba = _partial_predict_proba(model, X, k)
        losses.append(log_loss(y, proba, labels=model.classes_))
    return losses


def _plot_loss(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    base_params: Dict,
    n_estimators_values: Sequence[int],
    title: str,
    ylabel: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for n_estimators in n_estimators_values:
        params = {**base_params, "n_estimators": n_estimators}
        model = CustomRandomForestClassifier(**params)
        model.fit(X_fit, y_fit)
        losses = _collect_loss_history(model, X_eval, y_eval)
        ax.plot(
            range(1, n_estimators + 1),
            losses,
        )

    ax.set_xlabel("Число базовых алгоритмов в ансамбле")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return ax


def plot_training_loss(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_params: Dict,
    n_estimators_values: Iterable[int] = (200, 500),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:

    return _plot_loss(
        X_eval=X_train,
        y_eval=y_train,
        X_fit=X_train,
        y_fit=y_train,
        base_params=base_params,
        n_estimators_values=tuple(n_estimators_values),
        title="Log-loss на обучении",
        ylabel="Log-loss (train)",
        ax=ax,
    )


def plot_test_loss(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_params: Dict,
    n_estimators_values: Iterable[int] = (200, 500),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:

    return _plot_loss(
        X_eval=X_test,
        y_eval=y_test,
        X_fit=X_train,
        y_fit=y_train,
        base_params=base_params,
        n_estimators_values=tuple(n_estimators_values),
        title="Log-loss на тесте",
        ylabel="Log-loss (test)",
        ax=ax,
    )
