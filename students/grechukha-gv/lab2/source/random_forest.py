from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest: бэггинг деревьев sklearn с bootstrap и OOB-оценкой.
    Важность OOB^j — падение OOB accuracy при перестановке признака j
    с усреднением по нескольким повторам и оценкой стандартного отклонения.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str | int | float | None = "sqrt",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self._in_bag_masks: list[np.ndarray] = []
        self._tree_class_maps: list[np.ndarray] = []
        self.classes_: np.ndarray = np.array([])
        self.oob_score_: float = float("nan")
        self.oob_decision_mask_: np.ndarray = np.array([], dtype=bool)
        self.feature_importances_: np.ndarray = np.array([])
        self.oob_permutation_importances_: np.ndarray = np.array([])
        self.oob_permutation_importances_std_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples, _ = X.shape
        self.classes_ = np.sort(np.unique(y))

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self._in_bag_masks = []
        self._tree_class_maps = []

        for _ in range(self.n_estimators):
            boot_idx = rng.randint(0, n_samples, size=n_samples)
            in_bag = np.bincount(boot_idx, minlength=n_samples) > 0
            self._in_bag_masks.append(in_bag)

            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=int(rng.randint(0, 2**31 - 1)),
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.estimators_.append(tree)
            self._tree_class_maps.append(self._build_class_map(tree))

        oob_pred, mask = self._oob_predictions(X)
        self.oob_decision_mask_ = mask
        self.oob_score_ = (
            float(np.mean(oob_pred[mask] == y[mask])) if mask.any() else float("nan")
        )

        self._compute_impurity_importances()
        return self

    def _build_class_map(self, tree: DecisionTreeClassifier) -> np.ndarray:
        """Возвращает массив длины n_classes_, где i-й элемент — индекс класса self.classes_[i]
        в tree.classes_ или -1, если этого класса в дереве нет."""
        n_classes = len(self.classes_)
        mapping = np.full(n_classes, -1, dtype=np.int64)
        tree_cls = tree.classes_
        for k, cls in enumerate(self.classes_):
            idx = np.flatnonzero(tree_cls == cls)
            if idx.size:
                mapping[k] = int(idx[0])
        return mapping

    def _accumulate_oob_votes(
        self, X: np.ndarray, oob_votes: np.ndarray, oob_counts: np.ndarray
    ) -> None:
        for tree, in_bag, cls_map in zip(
            self.estimators_, self._in_bag_masks, self._tree_class_maps
        ):
            oob_idx = np.nonzero(~in_bag)[0]
            if oob_idx.size == 0:
                continue
            proba = tree.predict_proba(X[oob_idx])
            for k, tree_k in enumerate(cls_map):
                if tree_k >= 0:
                    oob_votes[oob_idx, k] += proba[:, tree_k]
            oob_counts[oob_idx] += 1

    def _oob_predictions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(oob_pred, mask) — mask отмечает объекты, для которых нашлось хотя бы одно OOB-дерево"""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        oob_votes = np.zeros((n_samples, n_classes), dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int32)
        self._accumulate_oob_votes(X, oob_votes, oob_counts)
        mask = oob_counts > 0
        oob_pred = np.zeros(n_samples, dtype=self.classes_.dtype)
        if mask.any():
            rows = oob_votes[mask]
            rows = rows / np.maximum(rows.sum(axis=1, keepdims=True), 1e-12)
            oob_pred[mask] = self.classes_[np.argmax(rows, axis=1)]
        return oob_pred, mask

    def _compute_impurity_importances(self) -> None:
        if not self.estimators_:
            self.feature_importances_ = np.array([])
            return
        acc = np.zeros_like(self.estimators_[0].feature_importances_, dtype=np.float64)
        for tree in self.estimators_:
            acc += tree.feature_importances_
        self.feature_importances_ = acc / len(self.estimators_)

    def compute_oob_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 20,
        random_state: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """OOB^j: для каждого признака j повторяем n_repeats раз случайную перестановку
        столбца j, пересчитываем OOB accuracy теми же деревьями (без переобучения) и
        возвращаем (mean drop, std drop) по повторам.

        Усреднение по повторам — стандарт permutation_importance из sklearn:
        одна перестановка даёт квантованный шум порядка 1/|OOB|.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        oob_pred, mask = self._oob_predictions(X)
        n_features = X.shape[1]

        if not mask.any() or n_repeats <= 0:
            self.oob_permutation_importances_ = np.zeros(n_features, dtype=np.float64)
            self.oob_permutation_importances_std_ = np.zeros(n_features, dtype=np.float64)
            return (
                self.oob_permutation_importances_,
                self.oob_permutation_importances_std_,
            )

        baseline = float(np.mean(oob_pred[mask] == y[mask]))
        seed = random_state if random_state is not None else self.random_state
        rng = np.random.RandomState(seed)

        drops = np.zeros((n_features, n_repeats), dtype=np.float64)
        for j in range(n_features):
            original = X[:, j].copy()
            for r in range(n_repeats):
                X[:, j] = rng.permutation(original)
                pred_p, mask_p = self._oob_predictions(X)
                joint = mask & mask_p
                if joint.any():
                    drops[j, r] = baseline - float(
                        np.mean(pred_p[joint] == y[joint])
                    )
            X[:, j] = original

        self.oob_permutation_importances_ = drops.mean(axis=1)
        self.oob_permutation_importances_std_ = drops.std(axis=1)
        return (
            self.oob_permutation_importances_,
            self.oob_permutation_importances_std_,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Мягкое голосование: усреднение нормированных predict_proba деревьев"""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        votes = np.zeros((n_samples, n_classes), dtype=np.float64)
        counts = np.zeros((n_samples, n_classes), dtype=np.float64)
        for tree, cls_map in zip(self.estimators_, self._tree_class_maps):
            proba = tree.predict_proba(X)
            for k, tree_k in enumerate(cls_map):
                if tree_k >= 0:
                    votes[:, k] += proba[:, tree_k]
                    counts[:, k] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(counts > 0, votes / np.maximum(counts, 1.0), 0.0)
        row_sums = avg.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, avg / np.maximum(row_sums, 1e-12), 1.0 / n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
