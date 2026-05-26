import numpy as np


def _sigmoid(values):
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def _squared_error(y):
    if len(y) == 0:
        return 0.0
    mean_value = np.mean(y)
    return np.sum((y - mean_value) ** 2)


def _merge_values(left_value, right_value, left_weight, right_weight):
    return left_weight * left_value + right_weight * right_value


class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self.tree = self._grow_tree(X, y)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._predict_one(self.tree, x) for x in X])

    def _grow_tree(self, X, y, depth=0):
        node_value = float(np.mean(y)) if len(y) else 0.0

        if (
            len(y) < self.min_samples_split
            or depth >= self.max_depth
            or np.allclose(y, y[0])
        ):
            return {"leaf": node_value}

        best_feature, best_threshold, best_loss = self._best_split(X, y)
        current_loss = _squared_error(y)
        if best_feature is None or best_loss >= current_loss:
            return {"leaf": node_value}

        col = X[:, best_feature]
        valid = ~np.isnan(col)
        left_mask = valid & (col <= best_threshold)
        right_mask = valid & (col > best_threshold)

        n_left = int(left_mask.sum())
        n_right = int(right_mask.sum())
        total = n_left + n_right

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "q_left": n_left / total,
            "q_right": n_right / total,
            "left": self._grow_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._grow_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _best_split(self, X, y):
        best_loss = np.inf
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            col = X[:, feature_idx]
            valid = ~np.isnan(col)
            if valid.sum() < 2 * self.min_samples_leaf:
                continue

            values = np.unique(col[valid])
            if len(values) < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                left_mask = valid & (col <= threshold)
                right_mask = valid & (col > threshold)

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                loss = _squared_error(y[left_mask]) + _squared_error(y[right_mask])
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_loss

    def _predict_one(self, node, x):
        if "leaf" in node:
            return node["leaf"]

        value = x[node["feature"]]
        if np.isnan(value):
            left_value = self._predict_one(node["left"], x)
            right_value = self._predict_one(node["right"], x)
            return _merge_values(left_value, right_value, node["q_left"], node["q_right"])

        if value <= node["threshold"]:
            return self._predict_one(node["left"], x)
        return self._predict_one(node["right"], x)


class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.classes_ = None
        self.initial_score_ = None
        self.trees_ = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self._validate_params()

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostingClassifier supports only binary targets.")

        y_binary = (y == self.classes_[1]).astype(float)
        positive_rate = np.clip(np.mean(y_binary), 1e-6, 1 - 1e-6)
        self.initial_score_ = float(np.log(positive_rate / (1 - positive_rate)))
        self.trees_ = []

        raw_predictions = np.full(X.shape[0], self.initial_score_, dtype=float)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            probabilities = _sigmoid(raw_predictions)
            residuals = y_binary - probabilities

            train_idx = self._sample_indices(X.shape[0], rng)
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[train_idx], residuals[train_idx])
            update = tree.predict(X)
            raw_predictions += self.learning_rate * update
            self.trees_.append(tree)

        return self

    def predict_proba(self, X):
        raw_predictions = self._raw_predict(X)
        positive_proba = _sigmoid(raw_predictions)
        negative_proba = 1.0 - positive_proba
        return np.column_stack([negative_proba, positive_proba])

    def predict(self, X):
        positive_proba = self.predict_proba(X)[:, 1]
        return np.where(positive_proba >= 0.5, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "subsample": self.subsample,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _raw_predict(self, X):
        if self.initial_score_ is None:
            raise ValueError("Model is not fitted yet.")

        X = np.array(X, dtype=float)
        raw_predictions = np.full(X.shape[0], self.initial_score_, dtype=float)
        for tree in self.trees_:
            raw_predictions += self.learning_rate * tree.predict(X)
        return raw_predictions

    def _sample_indices(self, n_samples, rng):
        if self.subsample >= 1.0:
            return np.arange(n_samples)

        sample_size = max(1, int(n_samples * self.subsample))
        return rng.choice(n_samples, size=sample_size, replace=False)

    def _validate_params(self):
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in the interval (0, 1].")
