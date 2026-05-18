import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class CustomRandomForest(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=100, max_features='sqrt', bootstrap=True, random_state=None,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

        self.estimators_ = []
        self.oob_score_ = 0.0
        self.oob_error_ = 1.0
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        if self.max_features == 'sqrt':
            k = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            k = int(np.log2(n_features))
        else:
            k = self.max_features

        self.estimators_ = []
        oob_indices_list = []
        feature_subsets_list = []

        oob_votes = np.zeros((n_samples, len(self.classes_)))
        oob_counts = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            feat_idx = rng.choice(n_features, size=k, replace=False)
            feature_subsets_list.append(feat_idx)

            if self.bootstrap:
                sample_idx = rng.randint(0, n_samples, size=n_samples)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[sample_idx] = False
                oob_idx = np.where(oob_mask)[0]
                X_b, y_b = X[sample_idx][:, feat_idx], y[sample_idx]
            else:
                X_b, y_b = X[:, feat_idx], y
                oob_idx = np.array([], dtype=int)

            oob_indices_list.append(oob_idx)

            tree = DecisionTreeClassifier(
                random_state=rng.randint(0, 10000),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion
            )
            tree.fit(X_b, y_b)
            self.estimators_.append(tree)

            if len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx][:, feat_idx])
                for i, idx in enumerate(oob_idx):
                    oob_votes[idx, preds[i]] += 1
                    oob_counts[idx] += 1

        valid_mask = oob_counts > 0
        if np.any(valid_mask):
            oob_preds = np.argmax(oob_votes[valid_mask], axis=1)
            self.oob_score_ = accuracy_score(y[valid_mask], oob_preds)
            self.oob_error_ = 1.0 - self.oob_score_
        else:
            self.oob_error_ = 1.0

        self.feature_importances_ = np.zeros(n_features)
        if self.oob_error_ > 1e-6:
            for j in range(n_features):
                X_perm = X.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])

                perm_votes = np.zeros((n_samples, len(self.classes_)))
                perm_counts = np.zeros(n_samples)
                for t in range(self.n_estimators):
                    idx_oob = oob_indices_list[t]
                    f_idx = feature_subsets_list[t]
                    if len(idx_oob) == 0:
                        continue

                    preds = self.estimators_[t].predict(X_perm[idx_oob][:, f_idx])
                    for i, idx in enumerate(idx_oob):
                        perm_votes[idx, preds[i]] += 1
                        perm_counts[idx] += 1

                p_mask = perm_counts > 0
                perm_error = (1.0 - accuracy_score(y[p_mask], np.argmax(perm_votes[p_mask], axis=1))) if np.any(p_mask) else self.oob_error_

                self.feature_importances_[j] = (perm_error - self.oob_error_) / self.oob_error_ * 100
        else:
            self.feature_importances_ = np.zeros(n_features)

        self.oob_indices_list_ = oob_indices_list
        self.feature_subsets_list_ = feature_subsets_list
        return self

    def predict(self, X):
        preds = np.array(
            [tree.predict(X[:, self.feature_subsets_list_[i]]) for i, tree in enumerate(self.estimators_)]).T
        return np.array([np.bincount(row, minlength=len(self.classes_)).argmax() for row in preds])