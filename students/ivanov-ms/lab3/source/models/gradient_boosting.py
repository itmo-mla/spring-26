import numpy as np
from sklearn.tree import DecisionTreeRegressor
from typing import Optional, Union
from pandas import DataFrame
import warnings

from utils.metrics import roc_auc, roc_curve


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier implementation following AdaBoost logic.

    Uses gradient descent in function space with decision trees as base learners.
    Optimizes binary cross-entropy loss for binary classification.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages/rounds to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between `learning_rate` and `n_estimators`.
    max_depth : int, default=3
        Maximum depth of each decision tree regressor.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    random_state : int or None, default=None
        Random seed for reproducibility.
    subsample : float, default=1.0
        Fraction of samples used for fitting each tree. If < 1.0, enables stochastic gradient boosting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
        subsample: float = 1.0,
        oob_score: bool = False
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.subsample = subsample
        self.oob_score = oob_score

        self.estimators_ = []
        self.classes_ = None
        self.n_features_ = None
        self.feature_names_ = None
        self.feature_importances_ = None

        self.train_loss_ = []
        self.train_auc_scores_ = []
        self.oob_scores_ = []

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.float64:
        return -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba)).mean()

    @staticmethod
    def _gradient(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
        return y_pred_proba - y_true

    @staticmethod
    def _hessian(y_true: np.ndarray, y_pred_proba: np.ndarray):
        return y_pred_proba * (1 - y_pred_proba)

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function (raw predictions) for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        raw_predictions : array of shape (n_samples,)
            Raw predictions (sum of tree predictions scaled by learning rate).
        """
        if not self.estimators_:
            raise ValueError("Estimator not fitted yet.")

        raw_predictions = np.zeros(X.shape[0])
        for estimator in self.estimators_:
            raw_predictions += self.learning_rate * estimator.predict(X)
        return raw_predictions

    def fit(self, X: Union[np.ndarray, DataFrame], y: np.ndarray) -> 'GradientBoostingClassifier':
        """
        Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values in {-1, 1}.

        Returns
        -------
        self : GradientBoostingClassifier
            Fitted estimator.
        """
        if isinstance(X, DataFrame):
            self.feature_names_ = list(X.columns)

        X = np.asarray(X)
        y = np.asarray(y).flatten()

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Target values must be in {0, 1}.")

        n_samples = X.shape[0]
        self.estimators_ = []
        self.train_loss_ = []
        self.train_auc_scores_ = []
        self.oob_scores_ = []

        # Initialize predictions to zero
        raw_predictions = np.zeros(n_samples)

        # Bootstrap sampling mask for OOB score computation
        oob_mask = np.ones(n_samples, dtype=bool)

        for i in range(self.n_estimators):
            seed = (self.random_state + i) if self.random_state is not None else None
            rng_i = np.random.RandomState(seed)

            # Subsample if needed
            if self.subsample < 1.0:
                sample_mask = rng_i.rand(n_samples) < self.subsample
                X_subset = X[sample_mask]
                y_subset = y[sample_mask]
                raw_subset = raw_predictions[sample_mask]
                # Update OOB mask
                oob_mask[sample_mask] = False
            else:
                X_subset = X
                y_subset = y
                raw_subset = raw_predictions
                oob_mask[:] = False

            # Compute probabilities from current raw predictions
            y_pred_proba = GradientBoostingClassifier._sigmoid(raw_subset)
            _gradient = GradientBoostingClassifier._gradient(y_subset, y_pred_proba)

            # Fit decision tree to negative gradient
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=seed
            )
            tree.fit(X_subset, -_gradient)

            # Add tree to ensemble
            self.estimators_.append(tree)

            # Update raw predictions (on full training set)
            raw_predictions += self.learning_rate * tree.predict(X)

            # Compute train loss and score at this iteration
            y_pred_proba = GradientBoostingClassifier._sigmoid(raw_predictions)
            loss = GradientBoostingClassifier._loss(y, y_pred_proba)
            train_acc = roc_auc(*roc_curve(y, y_pred_proba))
            self.train_loss_.append(loss)
            self.train_auc_scores_.append(train_acc)

            # Compute OOB score improvement if possible
            if not self.oob_score:
                continue

            if np.any(oob_mask):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    oob_pred = self.predict(X[oob_mask])
                    oob_acc = np.mean(oob_pred == y[oob_mask])
                    self.oob_scores_.append(oob_acc)

        # Compute feature importances
        self._compute_feature_importances()

        return self

    def _compute_feature_importances(self):
        """
        Compute feature importances as average importance from all estimators.
        """
        if not self.estimators_:
            self.feature_importances_ = np.zeros(self.n_features_)
            return

        total_importances = np.zeros(self.n_features_)
        for tree in self.estimators_:
            total_importances += tree.feature_importances_

        self.feature_importances_ = total_importances / len(self.estimators_)
        # Normalize to sum to 1
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        proba : array-like of shape (n_samples,) with probabilities of positive class
        """
        raw_pred = self._decision_function(X)
        pos_proba = GradientBoostingClassifier._sigmoid(raw_pred)
        return pos_proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        threshold: float
            Threshold for positive class

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels in {0, 1}.
        """
        proba = self.predict_proba(X)
        return np.where(proba >= threshold, 1, 0)

    def __str__(self):
        return f"{self.__class__.__name__}[" \
               f"n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, " \
               f"max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, " \
               f"subsample={self.subsample}]"
