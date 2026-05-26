from __future__ import annotations

import numpy as np
from scipy import sparse


class NmfModel:

    def __init__(self, *, n_components: int = 40, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.w_: np.ndarray | None = None
        self.h_: np.ndarray | None = None

    def fit(self, train_matrix: sparse.csr_matrix) -> "NmfModel":
        # Простая реализация NNMF через мультипликативные обновления (Lee-Seung)
        x = train_matrix.toarray().astype(np.float64)
        x = np.maximum(x, 0.0)

        n_users, n_items = x.shape
        k = min(self.n_components, min(n_users, n_items) - 1)
        rng = np.random.default_rng(self.random_state)

        w = rng.random((n_users, k)) + 1e-3
        h = rng.random((k, n_items)) + 1e-3

        eps = 1e-10
        for _ in range(300):
            # H <- H * (W^T X) / (W^T W H)
            numerator_h = w.T @ x
            denominator_h = (w.T @ w) @ h + eps
            h *= numerator_h / denominator_h

            # W <- W * (X H^T) / (W H H^T)
            numerator_w = x @ h.T
            denominator_w = w @ (h @ h.T) + eps
            w *= numerator_w / denominator_w

        self.w_ = w
        self.h_ = h
        return self

    def predict(self, user_matrix: sparse.csr_matrix) -> np.ndarray:
        if self.w_ is None or self.h_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        # Для простоты: используем только восстановление на обученной факторизации
        return self.w_ @ self.h_

    def predict_pairs(
        self,
        user_matrix: sparse.csr_matrix,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        return self.predict(user_matrix)[rows, cols]
