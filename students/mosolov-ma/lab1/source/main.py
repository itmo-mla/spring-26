import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tree import DecisionTree
import os
import logging
from datetime import datetime
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

LOGS_DIR = Path(__file__).parent.parent / "logs"



def setup_logger(log_dir=None):
    if log_dir is None:
        log_dir = LOGS_DIR
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"comparison_{timestamp}.log"
    log_path_relative = f"logs/{log_path.name}"
    
    logger = logging.getLogger("decision_tree_comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_path_relative


def count_nodes(node):
    """Подсчёт количества узлов в дереве"""
    if isinstance(node, Leaf):
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def count_nodes_sklearn(tree):
    """Подсчёт количества внутренних узлов в дереве sklearn"""
    return tree.tree_.node_count - tree.get_n_leaves()


def count_leaves(node):
    """Подсчёт количества листьев"""
    if isinstance(node, Leaf):
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def get_tree_depth(node, current_depth=0):
    """Вычисление глубины дерева"""
    if isinstance(node, Leaf):
        return current_depth
    return max(
        get_tree_depth(node.left, current_depth + 1),
        get_tree_depth(node.right, current_depth + 1)
    )


def print_tree_structure(node, indent=0, max_depth=5):
    """Визуализация структуры дерева"""
    if indent > max_depth:
        return
    if isinstance(node, Leaf):
        print("  " * indent + f"Leaf: class={node.value}")
    else:
        print("  " * indent + f"Node: X[{node.feature}] <= {node.threshold:.2f}")
        print("  " * indent + f"  [weights: L={node.left_weight:.2f}, R={node.right_weight:.2f}]")
        print_tree_structure(node.left, indent + 1, max_depth)
        print_tree_structure(node.right, indent + 1, max_depth)


def print_tree_structure_to_logger(node, logger, feature_names, node_index=None, max_nodes=10):
    """Визуализация структуры дерева в формате sklearn"""
    if node_index is None:
        node_index = [0]

    if node_index[0] >= max_nodes:
        return
    
    if isinstance(node, Leaf):
        logger.info(f"Leaf {node_index[0]}: class={node.value}")
        node_index[0] += 1
    else:
        feature_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
        logger.info(f"Node {node_index[0]}: {feature_name} <= {node.threshold:.4f}")
        node_index[0] += 1
        print_tree_structure_to_logger(node.left, logger, feature_names, node_index, max_nodes)
        print_tree_structure_to_logger(node.right, logger, feature_names, node_index, max_nodes)


def load_and_preprocess_data(missing_rate=0.1, random_state=42, logger=None):
    if logger is None:
        logger, _ = setup_logger()
    
    np.random.seed(random_state)
    
    logger.info("Загрузка датасета Heart Attack Prediction из Kaggle...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "nareshbhat/health-care-data-set-on-heart-attack-possibility",
        "heart.csv"
    )

    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df.columns = column_names
    
    logger.info(f"Всего образцов: {len(df)}")
    logger.info(f"Признаки: {column_names[:-1]}")
    logger.info(f"Целевая переменная: {column_names[-1]}")

    feature_cols = column_names[:-1]
    target_col = 'target'
    
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    
    logger.info(f"Распределение классов: 0={sum(y==0)}, 1={sum(y==1)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    X_train_nan = add_missing_values(X_train, missing_rate=missing_rate, random_state=random_state)
    X_test_nan = add_missing_values(X_test, missing_rate=missing_rate, random_state=random_state)
    
    logger.info(f"Добавлено пропусков в train: {sum(x is None for row in X_train_nan for x in row)}")
    logger.info(f"Добавлено пропусков в test: {sum(x is None for row in X_test_nan for x in row)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_nan, X_test_nan, feature_cols


def add_missing_values(X, missing_rate=0.1, random_state=42):
    """Добавление пропущенных значений (None)."""
    np.random.seed(random_state)
    X_with_nan = X.copy().astype(object)
    mask = np.random.rand(*X.shape) < missing_rate
    X_with_nan[mask] = None
    return X_with_nan


def main():
    logger, log_path = setup_logger(log_dir=LOGS_DIR)
    
    logger.info("=" * 70)
    logger.info("СРАВНЕНИЕ РЕАЛИЗАЦИИ DECISION TREE С SKLEARN")
    logger.info("Датасет: Heart Attack Prediction")
    logger.info("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_nan, X_test_nan, feature_cols = \
        load_and_preprocess_data(missing_rate=0.1, random_state=42, logger=logger)
    
    logger.info(f"\nПризнаки: {feature_cols}")
    
    # Custom
    custom_tree = DecisionTree(max_depth=5, min_samples_split=20, criterion='gini', min_gain=0.01)
    custom_tree.fit(X_train, y_train)
    
    # Custom с pruning
    custom_tree_pruned = DecisionTree(max_depth=5, min_samples_split=20, criterion='gini', min_gain=0.01)
    custom_tree_pruned.fit(X_train, y_train)
    custom_tree_pruned.prune(X_val, y_val)
    
    # Sklearn
    sklearn_tree = SklearnDecisionTree(
        max_depth=5, 
        min_samples_split=20, 
        criterion='gini',
        random_state=42
    )
    sklearn_tree.fit(X_train, y_train)
    
    # Sklearn с pruning
    sklearn_tree_pruned = SklearnDecisionTree(
        max_depth=5, 
        min_samples_split=20, 
        criterion='gini',
        ccp_alpha=0.01,
        random_state=42
    )
    sklearn_tree_pruned.fit(X_train, y_train)
    

    # ==================== СТАТИСТИКА ДЕРЕВЬЕВ ====================
    logger.info("\n" + "=" * 70)
    logger.info(" " * 25 + "СТАТИСТИКА ДЕРЕВЬЕВ")
    logger.info("=" * 70)
    
    # Количество узлов
    custom_nodes = count_nodes(custom_tree.tree)
    custom_pruned_nodes = count_nodes(custom_tree_pruned.tree)
    sklearn_nodes = count_nodes_sklearn(sklearn_tree)
    sklearn_pruned_nodes = count_nodes_sklearn(sklearn_tree_pruned)
    
    # Количество листьев
    custom_leaves = count_leaves(custom_tree.tree)
    custom_pruned_leaves = count_leaves(custom_tree_pruned.tree)
    sklearn_leaves = sklearn_tree.get_n_leaves()
    sklearn_pruned_leaves = sklearn_tree_pruned.get_n_leaves()
    
    # Глубина дерева
    custom_depth = get_tree_depth(custom_tree.tree)
    custom_pruned_depth = get_tree_depth(custom_tree_pruned.tree)
    sklearn_depth = sklearn_tree.get_depth()
    sklearn_pruned_depth = sklearn_tree_pruned.get_depth()
    
    logger.info(f"\n{'Параметр':<30} {'Custom':<12} {'Custom (pruned)':<16} {'Sklearn':<12} {'Sklearn (pruned)':<16}")
    logger.info("-" * 88)
    logger.info(f"{'Кол-во узлов решения:':<30} {custom_nodes:<12} {custom_pruned_nodes:<16} {sklearn_nodes:<12} {sklearn_pruned_nodes:<16}")
    logger.info(f"{'Кол-во листьев:':<30} {custom_leaves:<12} {custom_pruned_leaves:<16} {sklearn_leaves:<12} {sklearn_pruned_leaves:<16}")
    logger.info(f"{'Фактическая глубина:':<30} {custom_depth:<12} {custom_pruned_depth:<16} {sklearn_depth:<12} {sklearn_pruned_depth:<16}")
    

    # ==================== МЕТРИКИ КАЧЕСТВА ====================
    logger.info("\n" + "=" * 70)
    logger.info(" " * 25 + "МЕТРИКИ КАЧЕСТВА")
    logger.info("=" * 70)
    
    # Предсказания
    custom_pred = custom_tree.predict(X_test)
    custom_pruned_pred = custom_tree_pruned.predict(X_test)
    sklearn_pred = sklearn_tree.predict(X_test)
    sklearn_pruned_pred = sklearn_tree_pruned.predict(X_test)
    
    # Вычисление метрик
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1-score": f1_score,
    }
    
    logger.info(f"\n{'Метрика':<15} {'Custom':<12} {'Custom (pruned)':<16} {'Sklearn':<12} {'Sklearn (pruned)':<16}")
    logger.info("-" * 73)
    
    for name, func in metrics.items():
        custom_metric = func(y_test, custom_pred)
        custom_pruned_metric = func(y_test, custom_pruned_pred)
        sklearn_metric = func(y_test, sklearn_pred)
        sklearn_pruned_metric = func(y_test, sklearn_pruned_pred)
        
        logger.info(f"{name:<15} {custom_metric:<12.4f} {custom_pruned_metric:<14.4f} {sklearn_metric:<12.4f} {sklearn_pruned_metric:<16.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info(" " * 25 + "ТЕСТ С ПРОПУСКАМИ")
    logger.info("=" * 70)

    X_train_nan_np = np.array([[np.nan if x is None else x for x in row] for row in X_train_nan])
    X_test_nan_np = np.array([[np.nan if x is None else x for x in row] for row in X_test_nan])

    sklearn_tree_nan = SklearnDecisionTree(
        max_depth=5, 
        min_samples_split=20, 
        criterion='gini',
        random_state=42
    )
    sklearn_tree_nan.fit(X_train_nan_np, y_train)

    custom_tree_nan = DecisionTree(max_depth=5, min_samples_split=20, criterion='gini')
    custom_tree_nan.fit(X_train_nan, y_train)
    
    custom_nan_pred = custom_tree_nan.predict(X_test_nan)
    sklearn_nan_pred = sklearn_tree_nan.predict(X_test_nan_np)
    
    logger.info(f"\nCustom:")
    logger.info(f"   Accuracy: {accuracy_score(y_test, custom_nan_pred):.4f}")
    logger.info(f"   F1-score: {f1_score(y_test, custom_nan_pred):.4f}")
    
    logger.info(f"\nSklearn:")
    logger.info(f"   Accuracy: {accuracy_score(y_test, sklearn_nan_pred):.4f}")
    logger.info(f"   F1-score: {f1_score(y_test, sklearn_nan_pred):.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info(" " * 25 + "СТРУКТУРА ДЕРЕВЬЕВ")
    logger.info("=" * 70)
    
    logger.info("\n--- Custom дерево ---")
    print_tree_structure_to_logger(custom_tree.tree, logger, feature_cols)
    
    logger.info("\n--- Sklearn дерево ---")
    n_nodes = sklearn_tree.tree_.node_count
    for i in range(min(10, n_nodes)):
        feature = sklearn_tree.tree_.feature[i]
        threshold = sklearn_tree.tree_.threshold[i]
        if feature != -2:  # -2 означает лист
            logger.info(f"Node {i}: {feature_cols[feature]} <= {threshold:.4f}")
        else:
            value = sklearn_tree.tree_.value[i][0]
            logger.info(f"Leaf {i}: class={np.argmax(value)}")


    # ==================== ИТОГОВАЯ ТАБЛИЦА ====================
    logger.info("\n" + "=" * 70)
    logger.info(" " * 25 + "ИТОГОВОЕ СРАВНЕНИЕ")
    logger.info("=" * 70)

    acc_custom = accuracy_score(y_test, custom_pred)
    acc_sklearn = accuracy_score(y_test, sklearn_pred)
    f1_custom = f1_score(y_test, custom_pred)
    f1_sklearn = f1_score(y_test, sklearn_pred)

    logger.info(f"""
┌─────────────────────────────────────────────────────────────────┐
│ Параметр               │   Custom   │  Sklearn   │   Разница    │
├─────────────────────────────────────────────────────────────────┤
│ Узлов решения          │ {custom_nodes:>10} │ {sklearn_nodes:>10} │ {custom_nodes - sklearn_nodes:>+11}  │
│ Листьев                │ {custom_leaves:>10} │ {sklearn_leaves:>10} │ {custom_leaves - sklearn_leaves:>+11}  │
│ Глубина                │ {custom_depth:>10} │ {sklearn_depth:>10} │ {custom_depth - sklearn_depth:>+11}  │
│ Accuracy (test)        │ {acc_custom:>10.4f} │ {acc_sklearn:>10.4f} │ {acc_custom - acc_sklearn:>+11.4f}  │
│ F1-score (test)        │ {f1_custom:>10.4f} │ {f1_sklearn:>10.4f} │ {f1_custom - f1_sklearn:>+11.4f}  │
└─────────────────────────────────────────────────────────────────┘

Лог сохранён в: {log_path}
""")


if __name__ == "__main__":
    from tree import Leaf, Node
    main()
