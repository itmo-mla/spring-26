from __future__ import annotations

import numpy as np
from scipy import sparse


class SlimModel:
    def __init__(
        self,
        *,
        l1_coef: float = 0.5,
        l2_coef: float = 1.0,
        positive_only: bool = True,
        max_iter: int = 100,
    ) -> None:
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.positive_only = positive_only
        self.max_iter = max_iter
        self.item_similarity_: sparse.csr_matrix | None = None

    def fit(self, train_matrix: sparse.csr_matrix) -> "SlimModel":
        # SLIM с elastic-net (L1 + L2) через coordinate descent,
        # с той же параметризацией штрафов, что и в библиотечной реализации
        x = train_matrix.toarray().astype(np.float64)
        n_users, n_items = x.shape

        alpha = 2.0 * float(self.l2_coef) + float(self.l1_coef)
        l1_ratio = float(self.l1_coef) / alpha if alpha > 0 else 0.0
        # Под sklearn-параметризацию
        l1_penalty = alpha * l1_ratio
        l2_penalty = alpha * (1.0 - l1_ratio)

        eps = 1e-12
        w = np.zeros((n_items, n_items), dtype=np.float64)

        for j in range(n_items):
            X = x.copy()
            y = X[:, j].copy()
            X[:, j] = 0.0

            beta = np.zeros(n_items, dtype=np.float64)
            # residual r = y - X @ beta (starts as y)
            r = y.copy()

            col_norm2 = np.sum(X * X, axis=0)
            col_norm2[j] = 1.0

            for _ in range(self.max_iter):
                max_change = 0.0
                for i in range(n_items):
                    if i == j:
                        continue
                    Xi = X[:, i]
                    if col_norm2[i] <= eps:
                        continue

                    # sklearn использует усреднение по объектам (n)
                    rho = float(Xi.T @ (r + Xi * beta[i])) / max(n_users, 1)
                    new_beta = np.sign(rho) * max(abs(rho) - l1_penalty, 0.0) / (
                        (col_norm2[i] / max(n_users, 1)) + l2_penalty
                    )
                    if self.positive_only:
                        new_beta = max(new_beta, 0.0)

                    delta = new_beta - beta[i]
                    if abs(delta) > 0:
                        r -= Xi * delta
                        beta[i] = new_beta
                        max_change = max(max_change, abs(delta))

                if max_change < 1e-4:
                    break

            beta[j] = 0.0
            w[:, j] = beta

        self.item_similarity_ = sparse.csr_matrix(w)
        return self

    def predict(self, user_matrix: sparse.csr_matrix) -> np.ndarray:
        if self.item_similarity_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return (user_matrix @ self.item_similarity_).toarray()

    def predict_pairs(
        self,
        user_matrix: sparse.csr_matrix,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        return self.predict(user_matrix)[rows, cols]
