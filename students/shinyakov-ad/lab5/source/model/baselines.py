import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class KarypisSLIMReference:
    def __init__(
        self,
        algo="cd",
        nthreads=2,
        l1r=1.0,
        l2r=1.0,
    ):
        self.algo = algo
        self.nthreads = nthreads
        self.l1r = l1r
        self.l2r = l2r
        self.model_ = None
        self.pred_matrix_ = None

    def fit(self, dataset: dict):
        try:
            from SLIM import SLIM, SLIMatrix
        except ImportError as exc:
            raise NotImplementedError from exc
        
        matrix = dataset["train_matrix"]

        train_matrix = SLIMatrix(csr_matrix(matrix))
        params = {
            "algo": self.algo,
            "nthreads": self.nthreads,
            "l1r": self.l1r,
            "l2r": self.l2r,
        }
        self.model_ = SLIM()
        self.model_.train(params, train_matrix)
        coef = self.model_.to_csr()
        self.pred_matrix_ = matrix @ coef
        return self

    def predict_pairs(self, rows, cols):
        return np.asarray(self.pred_matrix_[rows, cols]).ravel()


class SklearnLatentSemanticModel:
    def __init__(self, n_components=12, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pred_matrix_ = None

    def fit(self, dataset: dict):
        matrix = dataset["train_matrix"]
        matrix = np.asarray(matrix, dtype=float)
        n_components = min(self.n_components, min(matrix.shape) - 1)
        model = TruncatedSVD(
            n_components=n_components,
            random_state=self.random_state,
        )
        reduced = model.fit_transform(matrix)
        self.pred_matrix_ = reduced @ model.components_
        return self

    def predict_pairs(self, rows, cols):
        return self.pred_matrix_[rows, cols]
