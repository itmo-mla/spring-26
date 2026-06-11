import os
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

class CustomALS:
    def __init__(self, factors, lmbda, epochs):
        self.factors = factors
        self.lmbda = lmbda
        self.epochs = epochs
        self.P = None
        self.Q = None

    def fit(self, R):
        n_users, n_items = R.shape
        self.P = np.random.rand(n_users, self.factors)
        self.Q = np.random.rand(n_items, self.factors)

        for _ in range(self.epochs):
            for u in range(n_users):
                self.P[u] = np.linalg.solve(self.Q.T @ self.Q + self.lmbda * np.eye(self.factors), self.Q.T @ R[u].T)
            for i in range(n_items):
                self.Q[i] = np.linalg.solve(self.P.T @ self.P + self.lmbda * np.eye(self.factors), self.P.T @ R[:, i])

    def predict(self):
        return self.P @ self.Q.T

class ReferenceALS:
    def __init__(self, factors, regularization, iterations):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42
        )

    def fit(self, R):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        R_sparse = csr_matrix(R)
        self.model.fit(R_sparse)

    def predict(self):
        U = self.model.user_factors
        V = self.model.item_factors

        if hasattr(U, 'to_numpy'):
            U = U.to_numpy()
        if hasattr(V, 'to_numpy'):
            V = V.to_numpy()

        return U @ V.T
