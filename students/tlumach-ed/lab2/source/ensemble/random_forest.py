import numpy as np
from tree.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=5, max_features="sqrt", eps_train=0.5, eps_oob=0.35):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_weights = None

        self.eps_train = eps_train
        self.eps_oob = eps_oob

        self.trees = []
        self.oob_indices = []
        self.accepted_trees_count = 0


    def fit(self, X, y):
        self.trees = []
        self.oob_indices = []
        self.accepted_trees_count = 0

        n_samples, n_features = X.shape

        for _ in range(self.n_trees):
            # bootstrap
            indices = np.random.choice(n_samples, n_samples, replace=True)
            oob = list(set(range(n_samples)) - set(indices))

            X_sample = X[indices]
            y_sample = y[indices]


            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )

            tree.fit(X_sample, y_sample)

            # Q(b_t, U_t) — ошибка на обучающей подвыборке
            train_preds = tree.predict(X_sample)
            train_error = 1.0 - np.mean(train_preds == y_sample)

            # Q(b_t, X^ℓ\U_t) — ошибка на OOB-сэмплах
            if len(oob) > 0:
                oob_preds = tree.predict(X[oob])
                oob_error = 1.0 - np.mean(oob_preds == y[oob])
            else:
                oob_error = 0.0

            # Проверка порогов
            if train_error > self.eps_train:
                continue
            if oob_error > self.eps_oob:
                continue

            self.trees.append(tree)
            self.oob_indices.append(oob)
            self.accepted_trees_count += 1

        self._compute_tree_weights(X, y)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])

        if self.tree_weights is None:
            weights = np.ones(len(self.trees)) / len(self.trees)
        else:
            weights = self.tree_weights

        return np.tensordot(weights, tree_probs, axes=1)

    def oob_score(self, X, y):
        preds = [[] for _ in range(len(y))]

        for tree, oob_idx in zip(self.trees, self.oob_indices):
            for i in oob_idx:
                pred = tree.predict([X[i]])[0]
                preds[i].append(pred)

        final_preds = []
        for p in preds:
            if len(p) == 0:
                final_preds.append(0)
            else:
                final_preds.append(np.bincount(p).argmax())

        return np.mean(np.array(final_preds) == y)

    def feature_importance(self, X, y):
        baseline = self.oob_score(X, y)
        importances = []

        for j in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, j])

            score = self.oob_score(X_permuted, y)
            importance = (baseline - score) / baseline * 100
            importances.append(importance)

        return np.array(importances)

    def _compute_tree_weights(self, X, y):
        weights = []

        for tree, oob_idx in zip(self.trees, self.oob_indices):
            if len(oob_idx) == 0:
                weights.append(0)
                continue

            preds = tree.predict(X[oob_idx])
            acc = np.mean(preds == y[oob_idx])
            weights.append(acc)

        weights = np.array(weights)

        # защита от деления на 0
        if weights.sum() == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / weights.sum()

        self.tree_weights = weights