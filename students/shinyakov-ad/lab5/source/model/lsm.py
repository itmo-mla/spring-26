import numpy as np

class LSM:
    def __init__(
        self,
        n_factors=12,
        learning_rate=0.03,
        reg=0.02,
        n_epochs=80,
        random_state=42
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.user_factors_ = None
        self.item_factors_ = None
        self.pred_matrix_ = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, dataset: dict):
        matrix = dataset["train_matrix"]
        rows = dataset["train_rows"]
        columns = dataset["train_cols"]
        ratings = dataset["train_ratings"]

        n_users, n_items = matrix.shape

        P = self.rng.normal(0, 1 / np.sqrt(self.n_factors), size=(n_users, self.n_factors))
        Q = self.rng.normal(0, 1 / np.sqrt(self.n_factors), size=(n_items, self.n_factors))

        for epoch in range(self.n_epochs):

            indices = np.arange(len(ratings))
            self.rng.shuffle(indices)

            for idx in indices:
                u = rows[idx]
                i = columns[idx]
                r = ratings[idx]

                p_u = P[u].copy()
                q_i = Q[i].copy()

                prediction = p_u @ q_i
                error = r - prediction

                P[u] += self.learning_rate * (error * q_i - self.reg * p_u)
                Q[i] += self.learning_rate * (error * p_u - self.reg * q_i)

        self.user_factors_ = P.copy()
        self.item_factors_ = Q.copy()
        self.pred_matrix_ = P @ Q.T

        return self
        

    def predict_pairs(self, rows, cols):
        return self.pred_matrix_[rows, cols]
