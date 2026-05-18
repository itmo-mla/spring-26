"""Узел бинарного решающего дерева (лист или внутренняя вершина)"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TreeNode:
    """Лист: kind=='leaf'; внутренний узел: kind=='internal'"""

    kind: str  # "leaf" | "internal"
    n_samples: int = 0
    leaf_class: Optional[int] = None
    feature: Optional[str] = None
    split_type: Optional[str] = None  # "cat" | "num"
    value: Any = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    weight_left: float = 0.0
    weight_right: float = 0.0
    majority_class: Optional[int] = None

    def replace_with_leaf(self, leaf_class: int, n_samples: int) -> None:
        """REP: заменить поддерево на лист, сохранив число объектов обучения"""
        self.kind = "leaf"
        self.n_samples = n_samples
        self.leaf_class = leaf_class
        self.feature = None
        self.split_type = None
        self.value = None
        self.left = None
        self.right = None
        self.weight_left = 0.0
        self.weight_right = 0.0
        self.majority_class = None

    @property
    def is_leaf(self) -> bool:
        return self.kind == "leaf"
