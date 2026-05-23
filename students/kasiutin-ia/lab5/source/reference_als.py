import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


class ImplicitALS:
    def __init__(
        self,
        factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.P = None
        self.Q = None

    def fit(self, R):
        R = csr_matrix(R).astype(np.float32)
        confidence = R.copy()
        confidence.data = 1.0 + self.alpha * confidence.data

        self.model.fit(confidence)

        self.P = self.model.user_factors
        self.Q = self.model.item_factors
        return self

    def predict(self, R=None):
        return (self.model.user_factors @ self.model.item_factors.T).astype(np.float32)
