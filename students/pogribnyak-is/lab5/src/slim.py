"""SLIM: Sparse Linear Method for Top-N Recommendation.
   Ning & Karypis (2011). Each item's score is a sparse linear combination
   of other items' scores: r_hat = R * A, A >= 0, diag(A) = 0.
   Each column solved independently via ElasticNet.
"""
import numpy as np
from sklearn.linear_model import ElasticNet
from tqdm import tqdm


class SLIM:
    def __init__(self, alpha: float = 0.5, l1_ratio: float = 0.5,
                 max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.W: np.ndarray | None = None  # item-item weight matrix

    def fit(self, R: np.ndarray) -> "SLIM":
        """R: (n_users, n_items) dense matrix."""
        n_items = R.shape[1]
        self.W = np.zeros((n_items, n_items), dtype=np.float32)

        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=False,
            positive=True,
            max_iter=self.max_iter,
            tol=self.tol,
            selection="cyclic",
        )

        for j in tqdm(range(n_items), desc="SLIM fitting"):
            y = R[:, j]
            if y.sum() == 0:
                continue
            X = np.delete(R, j, axis=1)
            model.fit(X, y)
            coef = model.coef_
            self.W[:j, j] = coef[:j]
            self.W[j + 1:, j] = coef[j:]

        return self

    def predict(self, R: np.ndarray) -> np.ndarray:
        assert self.W is not None, "Call fit first"
        return R @ self.W

    def predict_rating(self, R: np.ndarray, user_id: int, item_id: int) -> float:
        scores = self.predict(R)
        return float(scores[user_id, item_id])
