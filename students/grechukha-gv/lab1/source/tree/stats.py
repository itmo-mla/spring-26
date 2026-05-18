"""Метрики структуры дерева: число узлов, листьев и глубина"""

from .node import TreeNode


def count_nodes(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def count_leaves(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def tree_depth(node: TreeNode) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def tree_structure_summary(node: TreeNode) -> dict:
    return {
        "nodes": count_nodes(node),
        "leaves": count_leaves(node),
        "depth": tree_depth(node),
    }
