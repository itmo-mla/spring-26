import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ============================================================
#  Загрузка и подготовка данных (Titanic)
# ============================================================

def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked -> Survived
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    df = df[features + [target]].copy()

    # кодирую категориальные признаки числами, NaN оставляю
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(embarked_map)

    X = df[features].values.astype(float)
    y = df[target].values.astype(int)

    # какие признаки категориальные (индексы)
    cat_features = [0, 1, 6]  # Pclass, Sex, Embarked
    return X, y, features, cat_features


# ============================================================
#  Узел дерева
# ============================================================

class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.is_categorical = False
        self.left = None
        self.right = None
        self.is_leaf = False
        self.label = None
        self.class_distribution = None
        self.n_samples = 0
        # доля объектов, ушедших влево (для пропусков)
        self.left_ratio = 0.5


# ============================================================
#  ID3 с критерием Джини
# ============================================================

class DecisionTreeID3:
    def __init__(self, max_depth=10, min_samples_leaf=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None
        self.cat_features = []

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _gini_gain(self, y, left_mask, right_mask):
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        if n_left == 0 or n_right == 0:
            return 0.0
        gini_parent = self._gini(y)
        gini_left = self._gini(y[left_mask])
        gini_right = self._gini(y[right_mask])
        return gini_parent - (n_left / n * gini_left + n_right / n * gini_right)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_is_cat = False

        for f in range(n_features):
            col = X[:, f]
            known_mask = ~np.isnan(col)
            if np.sum(known_mask) == 0:
                continue

            col_known = col[known_mask]
            y_known = y[known_mask]

            if f in self.cat_features:
                # категориальный: разбиваю по каждому значению
                unique_vals = np.unique(col_known)
                for val in unique_vals:
                    left_mask = known_mask & (col == val)
                    right_mask = known_mask & (col != val)
                    gain = self._gini_gain(y, left_mask, right_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = f
                        best_threshold = val
                        best_is_cat = True
            else:
                # числовой: сортирую уникальные значения, пробую пороги
                unique_vals = np.unique(col_known)
                if len(unique_vals) <= 1:
                    continue
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                # беру не больше 50 порогов для скорости
                if len(thresholds) > 50:
                    idx = np.linspace(0, len(thresholds) - 1, 50, dtype=int)
                    thresholds = thresholds[idx]
                for thr in thresholds:
                    left_mask = known_mask & (col <= thr)
                    right_mask = known_mask & (col > thr)
                    gain = self._gini_gain(y, left_mask, right_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = f
                        best_threshold = thr
                        best_is_cat = False

        return best_feature, best_threshold, best_is_cat, best_gain

    def _build_tree(self, X, y, depth):
        node = Node()
        node.n_samples = len(y)

        classes, counts = np.unique(y, return_counts=True)
        node.label = classes[np.argmax(counts)]
        dist = np.zeros(self.n_classes)
        for c, cnt in zip(classes, counts):
            dist[c] = cnt
        node.class_distribution = dist / len(y)

        # условия остановки
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(classes) == 1):
            node.is_leaf = True
            return node

        feature, threshold, is_cat, gain = self._best_split(X, y)

        if feature is None or gain <= 0:
            node.is_leaf = True
            return node

        node.feature = feature
        node.threshold = threshold
        node.is_categorical = is_cat

        col = X[:, feature]
        known_mask = ~np.isnan(col)

        if is_cat:
            left_mask = known_mask & (col == threshold)
            right_mask = known_mask & (col != threshold)
        else:
            left_mask = known_mask & (col <= threshold)
            right_mask = known_mask & (col > threshold)

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            node.is_leaf = True
            return node

        # пропущенные значения распределяю пропорционально
        missing_mask = np.isnan(col)
        node.left_ratio = n_left / (n_left + n_right) if (n_left + n_right) > 0 else 0.5

        # объекты с пропусками отправляю в обе ветки пропорционально
        # (для обучения кладу их в большую ветку)
        if np.sum(missing_mask) > 0:
            if n_left >= n_right:
                left_mask = left_mask | missing_mask
            else:
                right_mask = right_mask | missing_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.class_distribution

        val = x[node.feature]

        if np.isnan(val):
            # пропуск: иду в обе ветки с весами
            left_dist = self._predict_one(x, node.left)
            right_dist = self._predict_one(x, node.right)
            return node.left_ratio * left_dist + (1.0 - node.left_ratio) * right_dist

        if node.is_categorical:
            if val == node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
        else:
            if val <= node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)

    def predict(self, X):
        predictions = []
        for x in X:
            dist = self._predict_one(x, self.root)
            predictions.append(np.argmax(dist))
        return np.array(predictions)

    def _tree_depth(self, node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))

    def depth(self):
        return self._tree_depth(self.root)

    def _count_nodes(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def n_nodes(self):
        return self._count_nodes(self.root)


# ============================================================
#  Редукция дерева (Reduced Error Pruning)
# ============================================================

def prune_tree(tree, X_val, y_val):
    def _accuracy_with_subtree(node, X, y):
        preds = []
        for x in X:
            dist = tree._predict_one(x, node)
            preds.append(np.argmax(dist))
        preds = np.array(preds)
        return np.mean(preds == y) if len(y) > 0 else 1.0

    def _get_samples_at_node(node, X, y):
        # собираю индексы объектов, дошедших до узла
        indices = []
        for i, x in enumerate(X):
            if _reaches_node(x, tree.root, node):
                indices.append(i)
        if len(indices) == 0:
            return np.empty((0, X.shape[1])), np.empty(0, dtype=int)
        indices = np.array(indices)
        return X[indices], y[indices]

    def _reaches_node(x, current, target):
        if current is target:
            return True
        if current.is_leaf:
            return False

        val = x[current.feature]
        if np.isnan(val):
            return (_reaches_node(x, current.left, target) or
                    _reaches_node(x, current.right, target))
        if current.is_categorical:
            if val == current.threshold:
                return _reaches_node(x, current.left, target)
            else:
                return _reaches_node(x, current.right, target)
        else:
            if val <= current.threshold:
                return _reaches_node(x, current.left, target)
            else:
                return _reaches_node(x, current.right, target)

    def _prune_recursive(node):
        if node is None or node.is_leaf:
            return

        _prune_recursive(node.left)
        _prune_recursive(node.right)

        # пробую заменить поддерево листом
        X_node, y_node = _get_samples_at_node(node, X_val, y_val)
        if len(y_node) == 0:
            return

        acc_before = _accuracy_with_subtree(node, X_node, y_node)

        # сохраняю
        old_left = node.left
        old_right = node.right
        old_is_leaf = node.is_leaf

        node.is_leaf = True
        acc_after = np.mean(np.argmax(node.class_distribution) == y_node)

        if acc_after >= acc_before:
            # редукция не ухудшает качество - оставляю лист
            node.left = None
            node.right = None
        else:
            # откатываю
            node.is_leaf = old_is_leaf
            node.left = old_left
            node.right = old_right

    _prune_recursive(tree.root)
    return tree


# ============================================================
#  Основной скрипт
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  Лабораторная работа №1. Логическая классификация")
    print("  Бинарное решающее дерево ID3 с критерием Джини")
    print("=" * 60)

    # загружаю данные
    print("\n--- Загрузка датасета Titanic ---")
    X, y, feature_names, cat_features = load_titanic()
    print(f"Размер датасета: {X.shape[0]} объектов, {X.shape[1]} признаков")
    n_missing = np.sum(np.isnan(X))
    print(f"Количество пропусков: {n_missing}")
    print(f"Признаки: {feature_names}")
    print(f"Категориальные: {[feature_names[i] for i in cat_features]}")

    # разбиваю на train/val/test (60/20/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    print(f"\nТренировочная выборка: {len(y_train)}")
    print(f"Валидационная выборка: {len(y_val)}")
    print(f"Тестовая выборка:      {len(y_test)}")

    # обучаю мое дерево
    print("\n--- Обучение дерева ID3 ---")
    my_tree = DecisionTreeID3(max_depth=10, min_samples_leaf=5, min_samples_split=10)
    my_tree.cat_features = cat_features

    t0 = time.time()
    my_tree.fit(X_train, y_train)
    my_time = time.time() - t0

    y_pred = my_tree.predict(X_test)
    my_acc = accuracy_score(y_test, y_pred)
    my_f1 = f1_score(y_test, y_pred)

    print(f"Глубина дерева:   {my_tree.depth()}")
    print(f"Количество узлов: {my_tree.n_nodes()}")
    print(f"Accuracy:         {my_acc:.4f}")
    print(f"F1-score:         {my_f1:.4f}")
    print(f"Время обучения:   {my_time:.4f} сек")

    # редукция дерева
    print("\n--- Редукция дерева ---")
    depth_before = my_tree.depth()
    nodes_before = my_tree.n_nodes()

    prune_tree(my_tree, X_val, y_val)

    y_pred_pruned = my_tree.predict(X_test)
    pruned_acc = accuracy_score(y_test, y_pred_pruned)
    pruned_f1 = f1_score(y_test, y_pred_pruned)

    print(f"Глубина до редукции:  {depth_before}, после: {my_tree.depth()}")
    print(f"Узлов до редукции:    {nodes_before}, после: {my_tree.n_nodes()}")
    print(f"Accuracy после:       {pruned_acc:.4f}")
    print(f"F1-score после:       {pruned_f1:.4f}")

    # sklearn для сравнения
    print("\n--- Эталон: sklearn DecisionTreeClassifier ---")
    # заполняю пропуски медианами для sklearn
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()
    for j in range(X.shape[1]):
        col = X_train_filled[:, j]
        median_val = np.nanmedian(col)
        X_train_filled[np.isnan(X_train_filled[:, j]), j] = median_val
        X_test_filled[np.isnan(X_test_filled[:, j]), j] = median_val

    sk_tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
    t0 = time.time()
    sk_tree.fit(X_train_filled, y_train)
    sk_time = time.time() - t0

    y_pred_sk = sk_tree.predict(X_test_filled)
    sk_acc = accuracy_score(y_test, y_pred_sk)
    sk_f1 = f1_score(y_test, y_pred_sk)

    print(f"Глубина дерева:   {sk_tree.get_depth()}")
    print(f"Accuracy:         {sk_acc:.4f}")
    print(f"F1-score:         {sk_f1:.4f}")
    print(f"Время обучения:   {sk_time:.4f} сек")

    # итоговая таблица
    print("\n" + "=" * 60)
    print("  Сравнение результатов")
    print("=" * 60)
    print(f"{'Метрика':<25} {'Мое дерево':<15} {'После редукции':<18} {'sklearn':<15}")
    print("-" * 73)
    print(f"{'Accuracy':<25} {my_acc:<15.4f} {pruned_acc:<18.4f} {sk_acc:<15.4f}")
    print(f"{'F1-score':<25} {my_f1:<15.4f} {pruned_f1:<18.4f} {sk_f1:<15.4f}")
    print(f"{'Глубина':<25} {depth_before:<15} {my_tree.depth():<18} {sk_tree.get_depth():<15}")
    print(f"{'Узлов':<25} {nodes_before:<15} {my_tree.n_nodes():<18} {sk_tree.tree_.node_count:<15}")
    print(f"{'Время обучения (сек)':<25} {my_time:<15.4f} {'':<18} {sk_time:<15.4f}")

    # ============================================================
    #  Графики
    # ============================================================

    # 1. распределение целевой переменной
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = np.bincount(y)
    ax.bar(['Погиб (0)', 'Выжил (1)'], counts, color=['#d62728', '#2ca02c'])
    for i, c in enumerate(counts):
        ax.text(i, c + 5, str(c), ha='center', fontsize=11)
    ax.set_title('Распределение целевой переменной (Titanic)')
    ax.set_ylabel('Количество объектов')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'target_distribution.png'), dpi=100)
    plt.close()

    # 2. пропуски по признакам
    fig, ax = plt.subplots(figsize=(8, 4))
    n_missing_per = np.isnan(X).sum(axis=0)
    bars = ax.bar(feature_names, n_missing_per, color='#1f77b4')
    for bar, val in zip(bars, n_missing_per):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2, str(int(val)),
                    ha='center', fontsize=10)
    ax.set_title('Количество пропусков по признакам')
    ax.set_ylabel('Пропусков')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'missing_values.png'), dpi=100)
    plt.close()

    # 3. сравнение метрик: моя / редуцированная / sklearn
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Моё ID3', 'После редукции', 'sklearn']
    accs = [my_acc, pruned_acc, sk_acc]
    f1s = [my_f1, pruned_f1, sk_f1]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, accs, w, label='Accuracy', color='#1f77b4')
    ax.bar(x + w / 2, f1s, w, label='F1-score', color='#ff7f0e')
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w / 2, a + 0.005, f'{a:.3f}', ha='center', fontsize=9)
        ax.text(i + w / 2, f + 0.005, f'{f:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Метрика')
    ax.set_title('Сравнение качества классификации')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'metrics_comparison.png'), dpi=100)
    plt.close()

    # 4. размер дерева до/после редукции
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ['Глубина', 'Количество узлов']
    before = [depth_before, nodes_before]
    after = [my_tree.depth(), my_tree.n_nodes()]
    sk = [sk_tree.get_depth(), sk_tree.tree_.node_count]
    x = np.arange(len(cats))
    w = 0.27
    ax.bar(x - w, before, w, label='До редукции', color='#1f77b4')
    ax.bar(x, after, w, label='После редукции', color='#2ca02c')
    ax.bar(x + w, sk, w, label='sklearn', color='#ff7f0e')
    for i, (b, a, s) in enumerate(zip(before, after, sk)):
        ax.text(i - w, b + 1, str(b), ha='center', fontsize=9)
        ax.text(i, a + 1, str(a), ha='center', fontsize=9)
        ax.text(i + w, s + 1, str(s), ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title('Размер дерева')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'tree_size.png'), dpi=100)
    plt.close()

    print(f"\nГрафики сохранены в {IMAGES_DIR}")
