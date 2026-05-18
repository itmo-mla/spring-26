from decision_tree import DecisionTreeClassifier
import numpy as np


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.oob_indices_ = []
        self.classes_ = None
        self.oob_score_ = None
        self.feature_importances_ = None
        self.rng = None

    def constrained_bootstrap(self, X, y):
        n_samples = len(y)

        bootstrap_indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        in_bootstrap = np.zeros(n_samples, dtype=bool)
        in_bootstrap[bootstrap_indices] = True
        oob_indices = np.where(~in_bootstrap)[0]

        return bootstrap_indices, oob_indices

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        self.rng = np.random.default_rng(self.random_state)
        self.trees = []
        self.oob_indices_ = []
        self.classes_ = np.unique(y)

        for _ in range(self.n_estimators):
            tree_max_depth = self.max_depth if self.max_depth is not None else np.inf
            tree_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
            tree = DecisionTreeClassifier(
                max_depth=tree_max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            bootstrap_indices, oob_indices = self.constrained_bootstrap(X, y)

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            self.oob_indices_.append(oob_indices)

        self.oob_score_ = self._calculate_oob_score(X, y)
        self.feature_importances_ = self._calculate_oob_feature_importances(X, y)
        return self

    def predict(self, X: np.ndarray):
        X = np.asarray(X)
        self._check_is_fitted()

        predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        return np.array([
            self._majority_vote(predictions[:, sample_idx])
            for sample_idx in range(X.shape[0])
        ])

    def predict_proba(self, X: np.ndarray):
        X = np.asarray(X)
        self._check_is_fitted()

        proba_sum = np.zeros((X.shape[0], len(self.classes_)))

        for tree in self.trees:
            tree_proba = tree.predict_proba(X)
            for tree_class_idx, class_label in enumerate(tree.classes_):
                class_idx = np.where(self.classes_ == class_label)[0][0]
                proba_sum[:, class_idx] += tree_proba[:, tree_class_idx]

        return proba_sum / len(self.trees)

    def _calculate_oob_score(self, X, y):
        y_pred = self._predict_oob(X)
        valid_mask = np.array([prediction is not None for prediction in y_pred])

        if valid_mask.sum() == 0:
            return np.nan

        return np.mean(y_pred[valid_mask] == y[valid_mask])

    def _predict_oob(self, X):
        X = np.asarray(X)
        self._check_is_fitted()

        votes = [[] for _ in range(X.shape[0])]

        for tree, oob_indices in zip(
            self.trees,
            self.oob_indices_,
        ):
            if len(oob_indices) == 0:
                continue

            predictions = tree.predict(X[oob_indices])
            for sample_idx, prediction in zip(oob_indices, predictions):
                votes[sample_idx].append(prediction)

        y_pred = np.empty(X.shape[0], dtype=object)
        y_pred[:] = None

        for sample_idx, sample_votes in enumerate(votes):
            if sample_votes:
                y_pred[sample_idx] = self._majority_vote(sample_votes)

        return y_pred

    def _calculate_oob_feature_importances(self, X, y):
        baseline_score = self.oob_score_
        if np.isnan(baseline_score):
            return np.full(X.shape[1], np.nan)

        importances = np.zeros(X.shape[1])

        for feature_idx in range(X.shape[1]):
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = self.rng.permutation(X_permuted[:, feature_idx])
            permuted_score = self._calculate_oob_score(X_permuted, y)
            importances[feature_idx] = baseline_score - permuted_score

        return importances

    def _check_is_fitted(self):
        if not self.trees:
            raise ValueError("RandomForestClassifier is not fitted yet")

    @staticmethod
    def _majority_vote(labels):
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]


def grid_search_oob(X, y, param_grid):
    from sklearn.model_selection import ParameterGrid

    best_model = None
    best_params = None
    best_score = -np.inf

    for params in ParameterGrid(param_grid):
        model = RandomForestClassifier(**params)
        model.fit(X, y)

        if model.oob_score_ > best_score:
            best_model = model
            best_params = params
            best_score = model.oob_score_

    return best_model, best_params, best_score