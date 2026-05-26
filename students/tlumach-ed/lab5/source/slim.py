import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import ElasticNet


class SLIM:
    def __init__(self, alpha=0.1, l1_ratio=0.5, max_iter=1000, positive=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.positive = positive
        self.W = None
        self.n_users = None
        self.n_items = None

    def fit(self, R):
        if issparse(R):
            R_dense = R.toarray().astype(np.float64)
        else:
            R_dense = np.array(R, dtype=np.float64)

        self.n_users, self.n_items = R_dense.shape
        rows, cols, vals = [], [], []

        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=False,
            max_iter=self.max_iter,
            positive=self.positive,
            copy_X=False,
            selection="cyclic",
        )


        for i in range(self.n_items):
            target = R_dense[:, i].copy()

            # Исключаем self-interaction: убираем i-й столбец из признаков
            features = R_dense.copy()
            features[:, i] = 0.0

            # Пропускаем пустые столбцы
            if target.sum() == 0:
                continue

            model.fit(features, target)
            coef = model.coef_
            coef[i] = 0.0     # исключаем self-interaction

            # Записываем только ненулевые веса
            nz = np.nonzero(coef)[0]
            for j in nz:
                rows.append(i)
                cols.append(j)
                vals.append(coef[j])

        # Собираем разреженную матрицу из накопленных ненулевых весов
        from scipy.sparse import coo_matrix
        self.W = coo_matrix(
            (vals, (rows, cols)),
            shape=(self.n_items, self.n_items),
            dtype=np.float32,
        ).tocsr()
        n_nonzero = self.W.nnz
        sparsity = 1.0 - n_nonzero / (self.n_items ** 2)
        print(f"Обучение завершено. Ненулевых весов: {n_nonzero:,}, "
              f"разреженность W: {sparsity:.4f}")
        return self

    def predict(self, R):
        if self.W is None:
            raise RuntimeError("Модель не обучена.")
        if issparse(R):
            R_pred = (R @ self.W.T).toarray()
        else:
            R_pred = np.array(R) @ self.W.T.toarray()
        return R_pred.astype(np.float32)

    def predict_df(self, test_df, train_R):
        if self.W is None:
            raise RuntimeError("Модель не обучена.")
        if issparse(train_R):
            R_pred = (train_R @ self.W.T).toarray()
        else:
            R_pred = np.array(train_R, dtype=np.float32) @ self.W.T.toarray()
        users = test_df["user_idx"].values
        items = test_df["item_idx"].values
        return R_pred[users, items].astype(np.float32)

    def predict_user(self, r_u):
        return self.W.dot(r_u).astype(np.float32)

    def recommend(self, R, user_idx, top_n=10, exclude_seen=True):
        if issparse(R):
            r_u = np.array(R[user_idx].todense()).flatten()
        else:
            r_u = R[user_idx]

        scores = self.predict_user(r_u)

        if exclude_seen:
            seen = np.where(r_u > 0)[0]
            scores[seen] = -np.inf

        top_items = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_items]