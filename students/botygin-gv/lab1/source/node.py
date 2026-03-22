import numpy as np
from typing import Optional


class TreeNode:
    def __init__(self):
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None
        self.is_leaf: bool = False
        self.class_label: Optional[int] = None
        self.class_distribution: Optional[np.ndarray] = None
        self.q_left: float = 0.0  # Вероятность ухода влево
        self.q_right: float = 0.0  # Вероятность ухода вправо
        self.samples_count: int = 0
