import numpy as np
from .node import Node


class DecisionTree:

    def __init__(self, max_depth=None, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes_ = None
        self.n_features_ = None
        self.tree_ = None

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if self.max_depth is None or depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                left_mask = X[:, idx] < thr
                right_mask = X[:, idx] >= thr

                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[right_mask], y[right_mask]

                node.feature_index = idx
                node.threshold = thr

                total = len(y_left) + len(y_right)

                node.q_left = len(y_left) / total
                node.q_right = len(y_right) / total

                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node


    def _best_split(self, X, y):
        m, n = X.shape

        if m < self.min_samples_split:
            return None, None

        best_gini = 1
        best_idx = None
        best_thr = None

        for idx in range(n):
            values = X[:, idx]
            # убираем NaN из поиска split
            mask = ~np.isnan(values)
            X_valid = values[mask]
            y_valid = y[mask]

            if len(X_valid) == 0:
                continue

            sorted_idx = np.argsort(X_valid)
            thresholds = X_valid[sorted_idx]

            for i in range(1, len(thresholds)):
                if thresholds[i] == thresholds[i - 1]:
                    continue
                thr = (thresholds[i] + thresholds[i - 1]) / 2

                valid = ~np.isnan(values)

                left_mask = valid & (values < thr)
                right_mask = valid & (values >= thr)

                y_left = y[left_mask]
                y_right = y[right_mask]

                gini = (
                    len(y_left) / m * self._gini(y_left)
                    + len(y_right) / m * self._gini(y_right)
                )

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr


    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        probs = [np.sum(y == c) / m for c in range(self.n_classes_)]

        return 1 - sum(p ** 2 for p in probs)


    def predict(self, X):
        probs = np.array([self._predict_proba(inputs, self.tree_) for inputs in X])
        return np.argmax(probs, axis=1)


    def _predict_proba(self, inputs, node):

        # если лист
        if node.left is None and node.right is None:
            counts = np.array(node.num_samples_per_class)
            return counts / counts.sum()

        val = inputs[node.feature_index]

        # если значение есть, то обычный спуск
        if not np.isnan(val):

            if val < node.threshold:
                return self._predict_proba(inputs, node.left)
            else:
                return self._predict_proba(inputs, node.right)

        # если NaN, то вероятностное распределение
        left_probs = self._predict_proba(inputs, node.left)
        right_probs = self._predict_proba(inputs, node.right)

        return (
                node.q_left * left_probs
                + node.q_right * right_probs
        )
