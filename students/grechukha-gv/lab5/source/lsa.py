import numpy as np
from scipy import sparse


class LatentSemanticAnalysis:
    """Latent semantic model based on a truncated singular value decomposition"""

    def __init__(self, n_components: int = 50) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def fit(self, matrix: sparse.spmatrix) -> "LatentSemanticAnalysis":
        x = self._validate_matrix(matrix).toarray()
        if self.n_components >= min(x.shape):
            raise ValueError("n_components must be smaller than both matrix dimensions")

        _, singular_values, components = np.linalg.svd(x, full_matrices=False)
        self.singular_values_ = singular_values[: self.n_components]
        self.components_ = components[: self.n_components]
        return self

    def transform(self, matrix: sparse.spmatrix) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("model is not fitted")
        x = self._validate_matrix(matrix)
        return np.asarray(x @ self.components_.T)

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("model is not fitted")
        return transformed @ self.components_

    def reconstruct(self, matrix: sparse.spmatrix) -> np.ndarray:
        return self.inverse_transform(self.transform(matrix))

    @staticmethod
    def _validate_matrix(matrix: sparse.spmatrix) -> sparse.csr_matrix:
        if not sparse.issparse(matrix):
            raise TypeError("matrix must be a scipy sparse matrix")
        x = matrix.tocsr().astype(float)
        if x.ndim != 2:
            raise ValueError("matrix must be a 2D matrix")
        return x
