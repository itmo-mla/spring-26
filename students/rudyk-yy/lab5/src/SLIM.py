import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class SLIM:

    def __init__(self, alpha=0.1, l1_ratio=0.5, max_iter=100, tol=1e-3, top_k=10):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.top_k = top_k
        self.W = None

    def fit(self, R):
        n_items = R.shape[1]
        self.W = np.zeros((n_items, n_items))

        sim = cosine_similarity(R.T) 

        for j in tqdm(range(n_items), desc='SLIM training'):
            r_j = R[:, j]
            if r_j.sum() == 0:
                continue

            # Take top_k neighbours by cosine similarity, excluding item j itself
            sim_j = sim[j].copy()
            sim_j[j] = -np.inf
            cols = np.argsort(sim_j)[-self.top_k:]

            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                positive=True,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=False,
            )
            model.fit(R[:, cols], r_j)
            self.W[cols, j] = model.coef_

        return self

    def predict_all(self, R):
        return R @ self.W

    def predict(self, R, user_idx, item_idx):
        return float(R[user_idx] @ self.W[:, item_idx])
