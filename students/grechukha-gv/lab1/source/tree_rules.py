"""Извлечение правил IF-THEN и частот сплитов по признакам"""

from tree import DecisionTreeID3
from tree.node import TreeNode


def collect_split_counts(node: TreeNode, counts=None):
    """Число сплитов по каждому признаку (до или после pruning — по переданному дереву)"""
    if counts is None:
        counts = {}
    if node.is_leaf:
        return counts
    feat = node.feature
    counts[feat] = counts.get(feat, 0) + 1
    collect_split_counts(node.left, counts)
    collect_split_counts(node.right, counts)
    return counts


def _format_leaf_line(conds, leaf_class, n_samples):
    body = " И ".join(conds) if conds else "(без условий)"
    return f"ЕСЛИ {body} -> класс {leaf_class} (объектов при обучении: {n_samples})"


def _dfs_collect_rules(node: TreeNode, conds, lines, max_rules):
    if len(lines) >= max_rules:
        return
    if node.is_leaf:
        lines.append(_format_leaf_line(conds, node.leaf_class, node.n_samples))
        return
    f_name = node.feature
    st = node.split_type
    val = node.value
    if st == "cat":
        left_c = conds + [f"{f_name} == {repr(val)}"]
        right_c = conds + [f"{f_name} != {repr(val)}"]
    else:
        left_c = conds + [f"{f_name} <= {val:g}"]
        right_c = conds + [f"{f_name} > {val:g}"]
    _dfs_collect_rules(node.left, left_c, lines, max_rules)
    _dfs_collect_rules(node.right, right_c, lines, max_rules)


def extract_rules(model: DecisionTreeID3, max_rules=14):
    """Список строк правил для обрезанного или полного дерева"""
    lines = []
    _dfs_collect_rules(model.tree, [], lines, max_rules)
    return lines


def tree_to_text(node: TreeNode, prefix="", is_last=True, lines=None):
    """Текстовое дерево с отступами (prefix)"""
    if lines is None:
        lines = []
    connector = "└── " if is_last else "├── "
    branch = prefix + connector if prefix else ""
    if node.is_leaf:
        lines.append(f"{branch}лист: класс {node.leaf_class}, n={node.n_samples}")
        return lines
    split_desc = f"{node.feature}"
    if node.split_type == "cat":
        split_desc += f" == {repr(node.value)} ?"
    else:
        split_desc += f" <= {node.value:g} ?"
    lines.append(f"{branch}{split_desc}")
    child_prefix = prefix + ("    " if is_last else "│   ")
    tree_to_text(node.left, child_prefix, False, lines)
    tree_to_text(node.right, child_prefix, True, lines)
    return lines
