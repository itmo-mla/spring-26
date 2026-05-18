import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def bagging(X, y):
    x_used = []
    y_used = []
    x_unused = []
    y_unused = []
    n = X.shape[0]
    used_indices = set()
    for i in range(n):
        index = random.randint(0, n - 1)
        x_used.append(X[index].copy())
        y_used.append(y[index].copy())
        used_indices.add(index)
    for index in range(n):
        if index not in used_indices:
            x_unused.append(X[index].copy())
            y_unused.append(y[index].copy())
    return np.array(x_used), np.array(y_used), np.array(x_unused), np.array(y_unused)


class _TreeMeta:
    def __init__(self, X_unused, y_unused):
        self.X_unused = X_unused
        self.y_unused = y_unused


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.clfs_ = []
        self.clfs_meta_ = []
        self._oob_score = None
        self.classes_ = np.unique(y)

        tree_kwargs = dict(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
        )

        for _ in range(self.n_estimators):
            clf = DecisionTreeClassifier(**tree_kwargs)
            X_t, y_t, X_oob, y_oob = bagging(X, y)
            clf.fit(X_t, y_t)
            self.clfs_.append(clf)
            self.clfs_meta_.append(_TreeMeta(X_oob, y_oob))
        return self

    def predict_proba(self, X):
        probs = self.clfs_[0].predict_proba(X)
        for i in range(1, len(self.clfs_)):
            probs += self.clfs_[i].predict_proba(X)
        return probs / len(self.clfs_)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def _calculate_oob(self):
        oob_values = []
        for i in range(len(self.clfs_)):
            preds = self.clfs_[i].predict(self.clfs_meta_[i].X_unused)
            oob_values.append(accuracy_score(self.clfs_meta_[i].y_unused, preds))
        self._oob_score = np.array(oob_values).mean()
        return np.array(oob_values)

    @property
    def oob_score_(self):
        if self._oob_score is None:
            self._calculate_oob()
        return self._oob_score

    def _calculate_feature_importance(self):
        oob_values = []
        for i in range(len(self.clfs_)):
            preds = self.clfs_[i].predict(self.clfs_meta_[i].X_unused)
            oob_values.append(accuracy_score(self.clfs_meta_[i].y_unused, preds))
        oob_score = np.array(oob_values).mean()

        X_unused = self.clfs_meta_[-1].X_unused
        y_unused = self.clfs_meta_[-1].y_unused
        n_features = X_unused.shape[1]

        feature_importances = []
        for f in range(n_features):
            X_copy = X_unused.copy()
            np.random.shuffle(X_copy[:, f])
            oob_shuffled = []
            for i in range(len(self.clfs_)):
                preds = self.clfs_[i].predict(X_copy)
                oob_shuffled.append(accuracy_score(y_unused, preds))
            oob_score_shuffled = np.array(oob_shuffled).mean()
            feature_importances.append(
                (oob_score_shuffled - oob_score) / oob_score
            )
        return feature_importances

    def score(self, X, y):
        self._calculate_oob()
        return self._oob_score
