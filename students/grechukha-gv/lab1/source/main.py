from data import artifacts_dir, load_chronic_kidney_disease
from plots import (
    plot_accuracy_vs_max_depth,
    plot_confusion_triple,
    plot_feature_importance,
    plot_metrics_bar,
)
from sklearn_baseline import (
    confusion_breakdown,
    metrics_report,
    print_metrics,
    run_sklearn_reference,
)
from tree import DecisionTreeID3, tree_structure_summary
from tree_rules import collect_split_counts, extract_rules, tree_to_text


def main():
    out_dir = artifacts_dir()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = (
        load_chronic_kidney_disease()
    )
    print(
        f"Выборки: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} "
        f"(Chronic Kidney Disease, UCI id=336)"
    )

    tree_raw = DecisionTreeID3(max_depth=None, min_samples_split=2)
    tree_raw.fit(X_train, y_train, feature_names)

    tree_pruned = DecisionTreeID3(max_depth=None, min_samples_split=2)
    tree_pruned.fit(X_train, y_train, feature_names)
    tree_pruned.prune(X_val, y_val)

    struct_raw = tree_structure_summary(tree_raw.tree)
    struct_pr = tree_structure_summary(tree_pruned.tree)
    print(
        "\nСтруктура дерева (узлы / листья / глубина): "
        f"до редукции {struct_raw['nodes']}/{struct_raw['leaves']}/{struct_raw['depth']}; "
        f"после {struct_pr['nodes']}/{struct_pr['leaves']}/{struct_pr['depth']}"
    )

    print("\nВалидация:")
    print_metrics("ID3", metrics_report(y_val, tree_raw.predict(X_val)))
    print_metrics("ID3 + pruning", metrics_report(y_val, tree_pruned.predict(X_val)))

    pred_raw_test = tree_raw.predict(X_test)
    pred_pr_test = tree_pruned.predict(X_test)

    m_raw = metrics_report(y_test, pred_raw_test)
    m_pr = metrics_report(y_test, pred_pr_test)
    sk_val, sk_test, pred_sk_val, pred_sk_test = run_sklearn_reference(
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    )

    print_metrics("sklearn (эталон, val)", sk_val)

    print("\nТест:")
    print_metrics("ID3 (до редукции)", m_raw)
    print_metrics("ID3 (после редукции)", m_pr)
    print_metrics("sklearn (эталон)", sk_test)

    pos = 1
    cb_raw = confusion_breakdown(y_test, pred_raw_test, positive_label=pos)
    cb_pr = confusion_breakdown(y_test, pred_pr_test, positive_label=pos)
    cb_sk = confusion_breakdown(y_test, pred_sk_test, positive_label=pos)
    print(
        "\nМатрица ошибок (положительный класс = CKD = 1): "
        "TN, FP, FN, TP на тесте"
    )
    print(f"  До редукции:    {cb_raw}")
    print(f"  После редукции: {cb_pr}")
    print(f"  sklearn:        {cb_sk}")

    results_test = {
        "ID3": m_raw,
        "ID3\npruned": m_pr,
        "sklearn": sk_test,
    }
    plot_metrics_bar(results_test, out_dir / "metrics_comparison.png")
    plot_confusion_triple(
        y_test, pred_raw_test, pred_pr_test, pred_sk_test, out_dir / "confusion_matrices.png"
    )
    plot_accuracy_vs_max_depth(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        out_dir / "accuracy_vs_depth.png",
        max_depth=12,
    )
    split_counts = collect_split_counts(tree_raw.tree)
    plot_feature_importance(split_counts, out_dir / "feature_importance.png")

    rules_lines = extract_rules(tree_pruned, max_rules=14)
    tree_txt = tree_to_text(tree_pruned.tree)
    rules_path = out_dir / "tree_rules.txt"
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write("Правила (до {} первых путей), дерево после редукции:\n".format(len(rules_lines)))
        f.write("\n".join(rules_lines))
        f.write("\n\n--- Текстовое дерево ---\n")
        f.write("\n".join(tree_txt))

    print(f"\nАртефакты сохранены в {out_dir}")


if __name__ == "__main__":
    main()
