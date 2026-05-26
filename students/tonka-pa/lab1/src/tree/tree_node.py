from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TreeNode:
    """Класс узла дерева. Не хватает двух плюсов и звездочек"""

    depth: int
    n_samples: int
    impurity: float
    prediction: Any
    leaf_risk: float
    class_counts: np.ndarray | None = None
    proba: np.ndarray | None = None
    feature_index: int | None = None
    feature_name: str | None = None
    feature_type: str | None = None
    threshold: float | None = None
    category: Any = None
    gain: float = 0.0
    q_left: float = 0.5
    q_right: float = 0.5
    n_known: int = 0
    n_left: int = 0
    n_right: int = 0
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def split_label(self) -> str:
        """Для отображения дерева на картинках."""
        if self.is_leaf or self.feature_name is None:
            return "leaf"
        if self.feature_type == "numeric":
            return f"{self.feature_name} <= {self.threshold:.6g}"
        return f"{self.feature_name} == {self.category!r}"

    def prune_to_leaf(self) -> None:
        """Сжимает поддерево в один лист."""
        self.left = None
        self.right = None
        self.feature_index = None
        self.feature_name = None
        self.feature_type = None
        self.threshold = None
        self.category = None
        self.gain = 0.0
        self.q_left = 0.5
        self.q_right = 0.5
        self.n_known = 0
        self.n_left = 0
        self.n_right = 0
