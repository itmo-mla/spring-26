import pandas as pd
from sklearn.model_selection import train_test_split

from utils.dataset import load_titanic
from utils.metrics import evaluate_model
from models.my_tree import DecisionTree, Pruner
from models.sklearn_tree import get_sklearn_tree


def make_report(rows, tree_stats):
    with open("data/report_template.md", "r", encoding="utf-8") as file:
        content = file.read()

    content = content.format(
        a = f'{rows[0]['accuracy']:.4}',
        b = f'{rows[0]['precision']:.4}',
        c = f'{rows[0]['recall']:.4}',
        d = f'{rows[0]['f1']:.4}',

        e = f'{rows[1]['accuracy']:.4}',
        f = f'{rows[1]['precision']:.4}',
        g = f'{rows[1]['recall']:.4}',
        h = f'{rows[1]['f1']:.4}',

        i = f'{rows[2]['accuracy']:.4}',
        j = f'{rows[2]['precision']:.4}',
        k = f'{rows[2]['recall']:.4}',
        l = f'{rows[2]['f1']:.4}',

        m = tree_stats['before_depth'],
        n = tree_stats['before_nodes'],
        o = tree_stats['before_leaves'],

        p = tree_stats['after_depth'],
        q = tree_stats['after_nodes'],
        r = tree_stats['after_leaves']
    )

    with open("data/report.md", "w", encoding="utf-8") as file:
        file.write(content)


def main():

    X_train, X_test, y_train, y_test, feature_types = load_titanic()

    sklearn_tree = get_sklearn_tree(X_train, y_train)

    my_tree = DecisionTree(
        feature_types=feature_types,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=5,
        min_gain=1e-4,
    )

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    my_tree.fit(X_train, y_train)

    stats_before = my_tree.get_stats()

    metrics_mytree_before = evaluate_model("My tree before pruning", my_tree, X_test, y_test)

    pruner = Pruner()

    pruner.reduced_error_prune(my_tree, X_val, y_val)

    stats_after = my_tree.get_stats()

    metrics_mytree_after = evaluate_model("My tree after pruning", my_tree, X_test, y_test)

    sklearn_result = evaluate_model("Sklearn DecisionTreeClassifier", sklearn_tree, X_test, y_test)

    make_report(
        rows=[metrics_mytree_before, metrics_mytree_after, sklearn_result],
        tree_stats={**stats_before, **stats_after},
    )


if __name__ == "__main__":
    main()