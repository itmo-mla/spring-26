import numpy as np
from scipy import sparse


class SlimRecommender:
    """Sparse Linear Method trained by projected proximal gradient descent"""

    def __init__(
        self,
        alpha_l1: float = 0.001,
        alpha_l2: float = 0.01,
        max_iter: int = 220,
        tol: float = 1e-5,
        random_state: int = 42,
    ) -> None:
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    def fit(self, interactions: sparse.spmatrix) -> "SlimRecommender":
        x = self._validate_interactions(interactions)
        gram = (x.T @ x).toarray() / x.shape[0]
        step = 1.0 / (self._estimate_lipschitz(gram) + self.alpha_l2)

        weights = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        previous_loss = np.inf

        for iteration in range(1, self.max_iter + 1):
            gradient = gram @ weights - gram + self.alpha_l2 * weights
            weights = self._proximal_non_negative_l1(weights - step * gradient, step * self.alpha_l1)
            np.fill_diagonal(weights, 0.0)

            loss = self._objective(gram, weights)
            self.loss_history_.append(loss)
            self.n_iter_ = iteration

            if abs(previous_loss - loss) < self.tol:
                break
            previous_loss = loss

        self.coef_ = weights
        return self

    def predict(self, interactions: sparse.spmatrix) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("model is not fitted")
        x = self._validate_interactions(interactions)
        return np.asarray(x @ self.coef_)

    def _objective(self, gram: np.ndarray, weights: np.ndarray) -> float:
        reconstruction = 0.5 * np.trace(weights.T @ gram @ weights) - np.trace(gram @ weights)
        ridge = 0.5 * self.alpha_l2 * np.sum(weights**2)
        sparsity = self.alpha_l1 * np.sum(weights)
        return float(reconstruction + ridge + sparsity)

    @staticmethod
    def _proximal_non_negative_l1(weights: np.ndarray, threshold: float) -> np.ndarray:
        return np.maximum(0.0, weights - threshold)

    def _estimate_lipschitz(self, gram: np.ndarray) -> float:
        rng = np.random.default_rng(self.random_state)
        vector = rng.normal(size=gram.shape[0])
        vector /= np.linalg.norm(vector)

        for _ in range(40):
            vector = gram @ vector
            norm = np.linalg.norm(vector)
            if norm == 0.0:
                return 1.0
            vector /= norm

        return float(vector @ gram @ vector)

    @staticmethod
    def _validate_interactions(interactions: sparse.spmatrix) -> sparse.csr_matrix:
        if not sparse.issparse(interactions):
            raise TypeError("interactions must be a scipy sparse matrix")
        matrix = interactions.tocsr().astype(float)
        if matrix.ndim != 2:
            raise ValueError("interactions must be a 2D matrix")
        if matrix.nnz == 0:
            raise ValueError("interactions matrix must contain non-zero values")
        return matrix
