import numpy as np

def prune_tree(tree, X_val, y_val):

    pruned_nodes = 0

    def _prune(node):
        nonlocal pruned_nodes

        if node.left is None or node.right is None:
            return

        _prune(node.left)
        _prune(node.right)

        if node.left and node.right:
            acc_before = accuracy(tree, X_val, y_val)

            left = node.left
            right = node.right

            node.left = None
            node.right = None

            acc_after = accuracy(tree, X_val, y_val)

            if acc_after < acc_before:
                node.left = left
                node.right = right
            else:
                pruned_nodes += 1

    _prune(tree.tree_)

    print(f"Pruned nodes: {pruned_nodes}")


def accuracy(tree, X, y):
    preds = tree.predict(X)
    return np.mean(preds == y)