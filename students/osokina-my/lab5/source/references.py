from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet


class LibrarySlimModel:
    def __init__(
        self,
        *,
        l1_coef: float = 0.5,
        l2_coef: float = 1.0,
        positive_only: bool = True,
        max_iter: int = 100,
    ) -> None:
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.positive_only = positive_only
        self.max_iter = max_iter
        self.item_similarity_: sparse.csr_matrix | None = None

    def fit(self, train_matrix: sparse.csr_matrix) -> "LibrarySlimModel":
        x_train = train_matrix.tocsc(copy=True)
        n_items = x_train.shape[1]

        alpha = 2.0 * self.l2_coef + self.l1_coef
        l1_ratio = self.l1_coef / alpha if alpha > 0 else 0.0
        regressor = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=self.positive_only,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=1e-4,
        )

        columns: list[np.ndarray] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for item_idx in range(n_items):
                target = x_train[:, item_idx].toarray().ravel()
                start = x_train.indptr[item_idx]
                end = x_train.indptr[item_idx + 1]
                backup = x_train.data[start:end].copy()
                x_train.data[start:end] = 0.0

                regressor.fit(x_train, target)
                columns.append(regressor.coef_.copy())
                x_train.data[start:end] = backup

        self.item_similarity_ = sparse.csr_matrix(np.column_stack(columns))
        return self

    def predict(self, user_matrix: sparse.csr_matrix) -> np.ndarray:
        if self.item_similarity_ is None:
            raise RuntimeError("Model is not fitted.")
        return (user_matrix @ self.item_similarity_).toarray()


class LibraryNmfModel:
    def __init__(self, *, n_components: int = 40, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.model_: NMF | None = None

    def fit(self, train_matrix: sparse.csr_matrix) -> "LibraryNmfModel":
        k = min(self.n_components, min(train_matrix.shape) - 1)
        self.model_ = NMF(
            n_components=k,
            init="nndsvda",
            solver="cd",
            random_state=self.random_state,
            max_iter=300,
        )
        self.model_.fit(train_matrix)
        return self

    def predict(self, user_matrix: sparse.csr_matrix) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.model_.inverse_transform(self.model_.transform(user_matrix))
