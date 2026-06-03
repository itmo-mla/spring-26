import numpy as np

class LFM:
    def __init__(
        self,
        n_factors=50,
        n_epochs=20,
        lr=0.005,
        reg_p=0.02,
        reg_q=0.02,
        reg_b=0.02,
        random_state=42,
        verbose=1,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_p = reg_p
        self.reg_q = reg_q
        self.reg_b = reg_b
        self.random_state = random_state
        self.verbose = verbose

        self.mu = 0.0
        self.bu = None
        self.bi = None
        self.P = None
        self.Q = None
        self.n_users = None
        self.n_items = None
        self.train_rmse_history = []

    def fit(self, train_df, n_users, n_items):
        rng = np.random.default_rng(self.random_state)

        self.n_users = n_users
        self.n_items = n_items
        self.mu = float(train_df["rating"].mean())

        # Инициализация параметров
        self.bu = np.zeros(n_users, dtype=np.float64)
        self.bi = np.zeros(n_items, dtype=np.float64)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        users = train_df["user_idx"].values
        items = train_df["item_idx"].values
        ratings = train_df["rating"].values.astype(np.float64)
        n_samples = len(ratings)

        print(f"Обучение LFM: {n_users} пользователей, {n_items} айтемов, "
              f"{self.n_factors} факторов, {n_samples:,} оценок")

        for epoch in range(1, self.n_epochs + 1):
            # Перемешиваем данные на каждой эпохе
            perm = rng.permutation(n_samples)
            users_e = users[perm]
            items_e = items[perm]
            ratings_e = ratings[perm]

            sq_err_sum = 0.0

            for k in range(n_samples):
                u = users_e[k]
                i = items_e[k]
                r = ratings_e[k]

                # Предсказание и ошибка
                r_hat = self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
                e = r - r_hat
                sq_err_sum += e * e

                # Обновление bias
                self.bu[u] += self.lr * (e - self.reg_b * self.bu[u])
                self.bi[i] += self.lr * (e - self.reg_b * self.bi[i])

                # Обновление латентных факторов
                # Сохраняем p_u до обновления чтобы не испортить обновление q_i
                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (e * self.Q[i] - self.reg_p * self.P[u])
                self.Q[i] += self.lr * (e * p_u_old - self.reg_q * self.Q[i])

            rmse = np.sqrt(sq_err_sum / n_samples)
            self.train_rmse_history.append(rmse)

        return self

    def predict_pair(self, user_idx, item_idx):
        return (self.mu
                + self.bu[user_idx]
                + self.bi[item_idx]
                + np.dot(self.P[user_idx], self.Q[item_idx]))

    def predict(self, test_df):
        users = test_df["user_idx"].values
        items = test_df["item_idx"].values

        preds = (self.mu
                 + self.bu[users]
                 + self.bi[items]
                 + np.sum(self.P[users] * self.Q[items], axis=1))
        return preds.astype(np.float32)

    def predict_all(self):
        return (self.mu
                + self.bu[:, np.newaxis]
                + self.bi[np.newaxis, :]
                + self.P @ self.Q.T).astype(np.float32)

    def recommend(self, user_idx, train_R, top_n=10, exclude_seen=True):
        scores = self.mu + self.bi + self.P[user_idx] @ self.Q.T

        if exclude_seen:
            from scipy.sparse import issparse
            if issparse(train_R):
                seen = train_R[user_idx].nonzero()[1]
            else:
                seen = np.where(train_R[user_idx] > 0)[0]
            scores[seen] = -np.inf

        top_items = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_items]