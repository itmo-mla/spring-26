import numpy as np 
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.min_samples_split=min_samples_split
        self.trees = []
        self.feature_importances_ = None
        self.n_indexes=[]
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices], indices
    def _oob_predict_single(self, x, i):
        oob_predictions = []
        for tree, indices in self.trees:
            if i not in indices:
                oob_predictions.append(tree.predict(x.reshape(1, -1))[0])
        if oob_predictions:
            return np.bincount(oob_predictions).argmax()
        else:
            return None
    def oob_score(self, X,y):
        preds = []
        for i in range(len(X)):
            pred = self._oob_predict_single(X[i], i)
            preds.append(pred)
        
        preds = np.array(preds)
        
        mask = np.not_equal(preds, None)
        if not np.any(mask):
            return 0.0
        preds = preds[mask]
        y = y[mask]
        return (preds==y).mean()
        

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        self.feature_importances_ = None
        for _ in range(self.n_estimators):
            X_sample, y_sample, indices = self._bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            oob_indices = np.setdiff1d(np.arange(len(X)), indices)
            if len(oob_indices) == 0 or (tree.predict(X[oob_indices]) == y[oob_indices]).mean() > 0.5:
                self.trees.append((tree, indices))

    def oob_feature_importance(self, X, y, random_state=None):
        baseline = self.oob_score(X, y)
        n_features = X.shape[1]
        importances = np.zeros(n_features, dtype=float)
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)

        for feature_idx in range(n_features):
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = rng.permutation(X_permuted[:, feature_idx])
            permuted_score = self.oob_score(X_permuted, y)
            importances[feature_idx] = baseline - permuted_score

        total = importances.sum()
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances
        return importances

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree, _ in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
    