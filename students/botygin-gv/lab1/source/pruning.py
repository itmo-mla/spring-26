import numpy as np
from tree import TreeNode, CustomDecisionTree


class TreePruner:
    def __init__(self, tree: CustomDecisionTree):
        self.tree = tree

    def _calculate_error(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(X) == 0:
            return 0.0
        preds = self.tree.predict(X)
        return np.mean(preds != y)

    def _prune_node(self, node: TreeNode, X_val: np.ndarray, y_val: np.ndarray) -> bool:
        if node.is_leaf:
            return False

        if node.left:
            self._prune_node(node.left, X_val, y_val)
        if node.right:
            self._prune_node(node.right, X_val, y_val)

        error_subtree = self._calculate_error(X_val, y_val)

        is_leaf_old = node.is_leaf
        left_old = node.left
        right_old = node.right
        class_label_old = node.class_label

        node.is_leaf = True
        node.left = None
        node.right = None

        error_leaf = self._calculate_error(X_val, y_val)

        if error_leaf <= error_subtree:
            return True
        else:
            node.is_leaf = is_leaf_old
            node.left = left_old
            node.right = right_old
            node.class_label = class_label_old
            return False

    def prune(self, X_val: np.ndarray, y_val: np.ndarray):
        self._prune_node(self.tree.root, X_val, y_val)
        return self.tree
