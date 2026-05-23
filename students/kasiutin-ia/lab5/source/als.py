import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


class ALS:
    def __init__(
        self,
        n_factors: int = 50,
        reg: float = 0.1,
        n_iter: int = 15,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iter = n_iter
        self.random_state = random_state

        self.P = None
        self.Q = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0

    def fit(self, R):
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)

        R = R.astype(np.float64)
        n_users, n_items = R.shape
        k = self.n_factors
        reg_eye = self.reg * np.eye(k, dtype=np.float64)

        self.global_mean = float(R.data.mean()) if R.nnz else 0.0

        rng = np.random.default_rng(self.random_state)
        self.P = (0.01 * rng.standard_normal((n_users, k))).astype(np.float64)
        self.Q = (0.01 * rng.standard_normal((n_items, k))).astype(np.float64)
        self.user_bias = np.zeros(n_users, dtype=np.float64)
        self.item_bias = np.zeros(n_items, dtype=np.float64)

        for _ in tqdm(range(self.n_iter), desc="ALS"):
            self._update_users(R, reg_eye)
            self._update_items(R, reg_eye)
            self._update_biases(R)

        return self

    def _update_users(self, R: csr_matrix, reg_eye: np.ndarray):
        for u in range(R.shape[0]):
            start, end = R.indptr[u], R.indptr[u + 1]
            if start == end:
                continue

            items = R.indices[start:end]
            ratings = R.data[start:end]
            Q_sub = self.Q[items]

            A = Q_sub.T @ Q_sub + reg_eye
            residuals = (
                ratings
                - self.global_mean
                - self.item_bias[items]
                - self.user_bias[u]
            )
            self.P[u] = np.linalg.solve(A, Q_sub.T @ residuals)

    def _update_items(self, R: csr_matrix, reg_eye: np.ndarray):
        R_t = R.T.tocsr()
        for i in range(R.shape[1]):
            start, end = R_t.indptr[i], R_t.indptr[i + 1]
            if start == end:
                continue

            users = R_t.indices[start:end]
            ratings = R_t.data[start:end]
            P_sub = self.P[users]

            A = P_sub.T @ P_sub + reg_eye
            residuals = (
                ratings
                - self.global_mean
                - self.user_bias[users]
                - self.item_bias[i]
            )
            self.Q[i] = np.linalg.solve(A, P_sub.T @ residuals)

    def _update_biases(self, R: csr_matrix):
        user_sum = np.zeros(R.shape[0], dtype=np.float64)
        user_cnt = np.zeros(R.shape[0], dtype=np.int64)
        item_sum = np.zeros(R.shape[1], dtype=np.float64)
        item_cnt = np.zeros(R.shape[1], dtype=np.int64)

        rows, cols = R.nonzero()
        preds = (
            self.global_mean
            + self.user_bias[rows]
            + self.item_bias[cols]
            + np.sum(self.P[rows] * self.Q[cols], axis=1)
        )
        residuals = R.data - preds

        np.add.at(user_sum, rows, residuals)
        np.add.at(user_cnt, rows, 1)
        np.add.at(item_sum, cols, residuals)
        np.add.at(item_cnt, cols, 1)

        mask_u = user_cnt > 0
        mask_i = item_cnt > 0
        self.user_bias[mask_u] += user_sum[mask_u] / user_cnt[mask_u]
        self.item_bias[mask_i] += item_sum[mask_i] / item_cnt[mask_i]

    def predict(self, R=None) -> np.ndarray:
        if self.P is None or self.Q is None:
            raise ValueError("Модель не обучена: сначала вызовите fit().")

        return (
            self.global_mean
            + self.user_bias[:, None]
            + self.item_bias[None, :]
            + self.P @ self.Q.T
        ).astype(np.float32)

    def recommend(
        self,
        user_idx: int,
        R_train: csr_matrix,
        top_k: int = 10,
        filter_seen: bool = True,
    ):
        scores = self.predict()[user_idx].copy()

        if filter_seen:
            seen = R_train[user_idx].indices
            scores[seen] = -np.inf

        top_items = np.argpartition(-scores, top_k)[:top_k]
        return top_items[np.argsort(-scores[top_items])]
