import numpy as np
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, List, Union, Any


class RandomForest:
    """
    Random Forest ensemble classifier

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    bootstrap : bool, default=True
        Whether to use bootstrap samples for training each tree.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization accuracy.
    random_state : int or None, default=None
        Random seed for reproducibility.
    tree_kwargs: Params for DecisionTreeClassifier. Default params:
        max_depth = None, min_samples_split = 2, max_features = 'sqrt'
        See more params on sklearn DecisionTreeClassifier documentation
    """
    DEFAULT_TREE_ARGS = {
        'max_depth': None,
        'min_samples_split': 2,
        'max_features': 'sqrt'
    }

    def __init__(
        self,
        n_estimators: int = 100,
        bootstrap_ratio: float = 0.632,
        oob_score: bool = True,
        random_state: Optional[int] = None,
        **tree_kwargs
    ):
        self.n_estimators = n_estimators
        self.bootstrap_ratio = bootstrap_ratio
        self.oob_score = oob_score
        self.random_state = random_state
        self.tree_kwargs = self.DEFAULT_TREE_ARGS.copy()
        self.tree_kwargs.update(tree_kwargs)

        self.estimators_: List[DecisionTreeClassifier] = []
        self.bootstrap_indices_: List[np.ndarray] = []
        self.oob_mask_ = None
        self.oob_score_ = None
        self.feature_names_ = None
        self.feature_importances_ = None
        self.oob_permutation_importance_ = None
        self.n_features_ = None
        self.classes_ = None

    def fit(self, X: Union[np.ndarray, DataFrame], y: np.ndarray) -> 'RandomForest':
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomForest
            Fitted estimator.
        """

        if isinstance(X, DataFrame):
            self.feature_names_ = list(X.columns)

        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.sort(np.unique(y))

        # Reset estimators and bootstrap_indices
        self.estimators_ = []
        self.bootstrap_indices_ = []

        for i in range(self.n_estimators):
            seed = (self.random_state + i) if self.random_state is not None else None
            rng_i = np.random.RandomState(seed)

            if self.bootstrap_ratio and 0 < self.bootstrap_ratio < 1:
                bootstrap_idx = rng_i.choice(n_samples, int(n_samples * self.bootstrap_ratio), replace=False)
            else:
                bootstrap_idx = np.arange(n_samples)

            tree = DecisionTreeClassifier(
                **self.tree_kwargs,
                random_state=seed
            )
            tree.fit(X[bootstrap_idx], y[bootstrap_idx])
            self.estimators_.append(tree)
            self.bootstrap_indices_.append(bootstrap_idx)

        if self.oob_score:
            if self.bootstrap_ratio and 0 < self.bootstrap_ratio < 1:
                self._compute_oob_score(X, y)
            else:
                print("WARNING: Bootstrap is disabled, OOB score won't be compute")

        self._compute_feature_importances()

        return self

    def _get_oob_predictions(self, X: np.ndarray):
        n_samples = X.shape[0]
        oob_votes = np.full((n_samples, len(self.classes_)), fill_value=0)
        oob_mask = np.zeros(n_samples, dtype=bool)
        full_range = np.arange(n_samples)

        for tree, b_idx in zip(self.estimators_, self.bootstrap_indices_):
            oob_idx = np.setdiff1d(full_range, b_idx)
            if len(oob_idx) == 0:
                continue
            preds = tree.predict(X[oob_idx])
            for idx, cls in enumerate(self.classes_):
                oob_votes[oob_idx[preds == cls], idx] += 1

            oob_mask[oob_idx] = True

        self.oob_mask_ = oob_mask

        oob_pred = self.classes_[np.argmax(oob_votes, axis=1)]
        return oob_pred

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray):
        """
        Compute out-of-bag predictions and score.
        """
        oob_preds = self._get_oob_predictions(X)
        self.oob_score_ = np.mean(oob_preds[self.oob_mask_] == y[self.oob_mask_])

    def _compute_feature_importances(self):
        """
        Average feature importances across all trees (impurity-based).
        """
        if not self.estimators_:
            return

        total_importances = np.zeros(self.n_features_)
        for tree in self.estimators_:
            total_importances += tree.feature_importances_
        self.feature_importances_ = total_importances / len(self.estimators_)

    def compute_oob_permutation_importance(self, X: np.ndarray, y: np.ndarray):
        """
        Compute permutation importance using OOB samples (OOB^j).

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data (must be same as used for fitting).
        y : array, shape (n_samples,)
            Target values.

        Returns
        -------
        importances : array, shape (n_features,)
            Mean decrease in OOB accuracy due to permutation.
        """
        if self.oob_mask_ is None:
            self._compute_oob_score(X, y)

        # Baseline OOB accuracy (only on samples with OOB predictions)
        baseline_acc = self.oob_score_

        X = np.asarray(X)
        n_samples, n_features = X.shape
        importances = np.zeros(n_features)

        for j in range(n_features):
            seed = (self.random_state + j) if self.random_state is not None else None
            rng_j = np.random.RandomState(seed)

            X_perm = X.copy()
            # Permute column j
            col = X_perm[:, j]
            rng_j.shuffle(col)

            y_perm_pred = self._get_oob_predictions(X_perm)
            perm_acc = np.mean(y_perm_pred[self.oob_mask_] == y[self.oob_mask_])
            importances[j] = baseline_acc - perm_acc

        self.oob_permutation_importance_ = importances
        return importances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        Majority vote across all trees.
        """
        X = np.asarray(X)

        votes = np.zeros((X.shape[0], len(self.classes_)), dtype=int)

        for tree in self.estimators_:
            preds = tree.predict(X)
            for idx, cls in enumerate(self.classes_):
                votes[preds == cls, idx] += 1

        y_pred = self.classes_[np.argmax(votes, axis=1)]
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        Average probability across all trees.
        Returns array of shape (n_samples, n_classes).
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        # Get probabilities from each tree
        probas = np.zeros((self.n_estimators, n_samples, len(self.classes_)))
        for idx, tree in enumerate(self.estimators_):
            probas[idx] = tree.predict_proba(X)
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
