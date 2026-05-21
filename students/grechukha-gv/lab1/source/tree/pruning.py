"""Reduced Error Pruning с маской объектов валидации, дошедших до узла."""

import numpy as np
import pandas as pd

from .node import TreeNode


def reduced_error_prune(model, X_val, y_val) -> None:
    """REP: обходит дерево и заменяет узлы на листья, если это не ухудшает ошибку на val"""
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    reach = np.ones(len(X_val), dtype=bool)
    _prune_node(model.tree, model, X_val, y_val, reach)


def _child_reach_masks(
    node: TreeNode, X_val: np.ndarray, model, reach_mask: np.ndarray
):
    n = len(X_val)
    left_m = np.zeros(n, dtype=bool)
    right_m = np.zeros(n, dtype=bool)
    if node.is_leaf:
        return left_m, right_m
    f_name = node.feature
    idx = model.feature_names.index(f_name)
    col = X_val[:, idx]
    for i in range(n):
        if not reach_mask[i]:
            continue
        val = col[i]
        if pd.isna(val):
            continue
        if node.split_type == "cat":
            if val == node.value:
                left_m[i] = True
            else:
                right_m[i] = True
        else:
            if float(val) <= float(node.value):
                left_m[i] = True
            else:
                right_m[i] = True
    return left_m, right_m


def _prune_node(
    node: TreeNode,
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    reach_mask: np.ndarray,
) -> None:
    if node.is_leaf:
        return

    left_rm, right_rm = _child_reach_masks(node, X_val, model, reach_mask)
    _prune_node(node.left, model, X_val, y_val, left_rm)
    _prune_node(node.right, model, X_val, y_val, right_rm)

    if not np.any(reach_mask):
        return

    Xm = X_val[reach_mask]
    ym = y_val[reach_mask]
    pred_before = model._predict_subtree(node, Xm)
    err_before = np.mean(pred_before != ym)

    leaf_class = node.majority_class
    pred_leaf = np.full(len(ym), leaf_class)
    err_leaf = np.mean(pred_leaf != ym)

    if err_leaf <= err_before:
        node.replace_with_leaf(leaf_class, node.n_samples)
