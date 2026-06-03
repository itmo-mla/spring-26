from sklearn.linear_model import ElasticNet
import numpy as np

class SLIM:
    def __init__(
        self,
        alpha=0.001,
        l1_ratio=0.1,
        max_iter=1000,
        positive=True,
        random_state=42,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.positive = positive
        self.random_state = random_state
        self.coef_ = None
        self.pred_matrix_ = None

    def fit(self, dataset: dict):

        matrix = dataset["train_matrix"]
        
        n_items = matrix.shape[1]
        self.coef_ = np.zeros((n_items, n_items), dtype=float)

        for item in range(n_items):

            y = matrix[:, item]
            X = matrix.copy()
            X[:, item] = 0

            elastic_net = ElasticNet(
                alpha = self.alpha,
                l1_ratio = self.l1_ratio,
                max_iter = self.max_iter,
                positive = self.positive,
                random_state = self.random_state,
                fit_intercept=False, 
                copy_X=False
            )

            elastic_net.fit(X, y)
            coef_j = elastic_net.coef_

            self.coef_[:, item] = coef_j
            self.coef_[item, item] = 0
        
        self.pred_matrix_ = matrix @ self.coef_

        return self

    def predict_pairs(self, rows, cols):
        return self.pred_matrix_[rows, cols]
