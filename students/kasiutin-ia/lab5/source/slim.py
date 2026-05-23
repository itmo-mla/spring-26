import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet
from tqdm import tqdm


class SLIM:
    def __init__(
        self,
        l1_reg=1e-3,
        l2_reg=1e-3,
        positive_only=True,
        fit_intercept=False,
        max_iter=100,
        tol=1e-4,
    ):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.positive_only = positive_only
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

        self.W = None

    def fit(self, R):
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)

        R = R.astype(np.float32)

        n_users, n_items = R.shape

        W = np.zeros((n_items, n_items), dtype=np.float32)

        alpha = self.l1_reg + self.l2_reg
        l1_ratio = self.l1_reg / alpha

        for item_idx in tqdm(range(n_items), total=n_items):

            y = R[:, item_idx].toarray().ravel()

            X = R.copy().tolil()
            X[:, item_idx] = 0
            X = X.tocsr()

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                positive=self.positive_only,
                fit_intercept=self.fit_intercept,
                copy_X=False,
                precompute=True,
                selection='cyclic',
                max_iter=self.max_iter,
                tol=self.tol,
            )

            model.fit(X, y)

            w = model.coef_

            w[item_idx] = 0.0

            W[:, item_idx] = w

        self.W = csr_matrix(W)

        return self

    def predict(self, R):
        return R @ self.W

    def recommend(self, user_vector, top_k=10, filter_seen=True):
        scores = user_vector @ self.W

        scores = np.asarray(scores).ravel()

        if filter_seen:
            scores[user_vector.nonzero()[1]] = -np.inf

        top_items = np.argpartition(-scores, top_k)[:top_k]

        return top_items[np.argsort(-scores[top_items])]