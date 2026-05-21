import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressor:
    def __init__(
        self,
        max_depth=3,
        min_samples_split=10,
        n_thresholds=20
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_thresholds = n_thresholds
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_row(x, self.root) for x in X])

    def _variance(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)

    # Friedman improvement
    def _gain(self, y, y_left, y_right):
        if len(y_left) == 0 or len(y_right) == 0:
            return -np.inf

        total_var = self._variance(y)

        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)

        child_var = (
            left_weight * self._variance(y_left) +
            right_weight * self._variance(y_right)
        )

        return total_var - child_var

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = -np.inf

        n_samples, n_features = X.shape

        for feature in range(n_features):
            feature_values = X[:, feature]

            thresholds = np.percentile(
                feature_values,
                np.linspace(5, 95, self.n_thresholds)
            )

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = self._gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.var(y) < 1e-8
        ):
            return Node(value=np.mean(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def _predict_row(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)

        return self._predict_row(x, node.right)
    