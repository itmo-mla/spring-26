"""
Лабораторная работа: Градиентный бустинг
Собственная реализация vs scikit-learn | Датасет: California Housing
"""

import time

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
TEST_SIZE = 0.2
N_FOLDS = 5
RANDOM_STATE = 42


class GradientBoosting:
    """Градиентный бустинг для регрессии (функция потерь — MSE)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self._trees: list[DecisionTreeRegressor] = []
        self._baseline: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        self._baseline = y.mean()
        self._trees = []

        F = np.full(len(y), self._baseline)
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, random_state=RANDOM_STATE
            )
            tree.fit(X, y - F)  # обучаем на псевдо-остатках
            F += self.learning_rate * tree.predict(X)
            self._trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = np.full(len(X), self._baseline)
        for tree in self._trees:
            result += self.learning_rate * tree.predict(X)
        return result


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Загружает и стандартизирует California Housing; возвращает train/test."""
    raw = fetch_california_housing()
    X = StandardScaler().fit_transform(raw.data)
    y = raw.target
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def _make_model(model_type: str) -> GradientBoosting | GradientBoostingRegressor:
    if model_type == "custom":
        return GradientBoosting(**PARAMS)
    return GradientBoostingRegressor(**PARAMS, random_state=RANDOM_STATE)


def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    """Обучает модель и возвращает метрики + время."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "time": elapsed,
    }


def cross_validate(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """5-fold CV для обеих моделей; возвращает mean ± std по каждой метрике."""
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {
        "custom": {"mse": [], "r2": [], "time": []},
        "sklearn": {"mse": [], "r2": [], "time": []},
    }

    for train_idx, val_idx in kfold.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        for model_type in ("custom", "sklearn"):
            metrics = evaluate(_make_model(model_type), X_tr, y_tr, X_val, y_val)
            for key, val in metrics.items():
                results[model_type][key].append(val)

    return {
        model_type: {
            key: (np.mean(vals), np.std(vals))
            for key, vals in metrics.items()
        }
        for model_type, metrics in results.items()
    }


def _header(title: str) -> None:
    width = 62
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_cv_results(cv: dict[str, dict]) -> None:
    _header(f"КРОСС-ВАЛИДАЦИЯ ({N_FOLDS}-fold)")
    labels = {"custom": "Моя реализация", "sklearn": "Scikit-learn   "}
    for model_type, label in labels.items():
        m = cv[model_type]
        print(f"\n  {label}")
        print(f"    MSE   = {m['mse'][0]:.4f} ± {m['mse'][1]:.4f}")
        print(f"    R²    = {m['r2'][0]:.4f} ± {m['r2'][1]:.4f}")
        print(f"    Время = {m['time'][0]:.3f} ± {m['time'][1]:.3f} сек")


def print_test_results(custom: dict, sklearn: dict) -> None:
    _header("СРАВНЕНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"\n  {'Метрика':<14} {'Моя реализация':>16} {'Scikit-learn':>14} {'Δ':>10}")
    print(f"  {'─' * 56}")

    def delta(a, b):
        return (a - b) / abs(b) * 100

    rows = [
        ("MSE", custom["mse"], sklearn["mse"], f"{delta(custom['mse'], sklearn['mse']):+.2f}%"),
        ("R²", custom["r2"], sklearn["r2"], f"{delta(custom['r2'], sklearn['r2']):+.2f}%"),
        ("Время, сек", custom["time"], sklearn["time"], f"{delta(custom['time'], sklearn['time']):+.2f}%"),
    ]
    for name, c_val, s_val, d in rows:
        fmt = ".4f" if name != "Время, сек" else ".3f"
        print(f"  {name:<14} {c_val:>16{fmt}} {s_val:>14{fmt}} {d:>10}")


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])

    cv = cross_validate(X_all, y_all)
    print_cv_results(cv)

    custom_metrics = evaluate(_make_model("custom"), X_train, y_train, X_test, y_test)
    sklearn_metrics = evaluate(_make_model("sklearn"), X_train, y_train, X_test, y_test)
    print_test_results(custom_metrics, sklearn_metrics)


if __name__ == "__main__":
    main()
