import numpy as np

_LOG_2PI = np.log(2.0 * np.pi)
_REG_EPS = 1e-4


def logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    cov_reg = cov + _REG_EPS * np.eye(d)
    L = np.linalg.cholesky(cov_reg)
    diff = X - mean
    solved = np.linalg.solve(L, diff.T)
    quad = np.sum(solved * solved, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * _LOG_2PI + log_det + quad)


def _kmeans_init(
    X: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    K = n_clusters
    centers = np.empty((K, d))

    idx = int(rng.integers(n))
    centers[0] = X[idx]
    for k in range(1, K):
        dist_sq = np.min(
            np.sum((X[:, None, :] - centers[:k][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = dist_sq.sum()
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=dist_sq / total))
        centers[k] = X[idx]

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[int(rng.integers(n))]

    weights = np.bincount(labels, minlength=K).astype(float) / n
    covariances = np.zeros((K, d, d))
    global_cov = np.cov(X, rowvar=False)
    if global_cov.ndim == 0:
        global_cov = np.array([[float(global_cov)]])
    for k in range(K):
        mask = labels == k
        if np.sum(mask) > d + 1:
            covariances[k] = np.cov(X[mask], rowvar=False)
        else:
            covariances[k] = global_cov.copy()
        covariances[k] += _REG_EPS * np.eye(d)

    return weights, centers, covariances


class MyGMM:
    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 200,
        tol: float = 1e-6,
        n_init: int = 5,
        random_state: int | None = 42,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.log_likelihood_history_: list[float] = []
        self.converged_iter_: int | None = None
        self.best_log_likelihood_: float = -np.inf
        self.n_iter_: int = 0

    def _initialize_params(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _kmeans_init(X, self.n_components, rng)

    def _e_step(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        n, K = X.shape[0], self.n_components
        log_resp = np.empty((n, K))
        for k in range(K):
            log_resp[:, k] = np.log(weights[k] + 1e-300) + _log_gaussian_pdf(
                X, means[k], covariances[k]
            )

        log_likelihood = float(np.sum(logsumexp(log_resp, axis=1)))
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        responsibilities = np.exp(log_resp - log_norm)
        return responsibilities, log_likelihood

    def _m_step(
        self, X: np.ndarray, responsibilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, d = X.shape
        K = self.n_components

        N_k = responsibilities.sum(axis=0)
        N_k = np.maximum(N_k, 1e-10)

        weights = N_k / n
        means = (responsibilities.T @ X) / N_k[:, np.newaxis]

        diff = X[:, np.newaxis, :] - means[np.newaxis, :, :]
        weighted_diff = responsibilities[:, :, np.newaxis] * diff
        covariances = np.einsum("nkd,nke->kde", weighted_diff, diff) / N_k[
            :, np.newaxis, np.newaxis
        ]
        covariances += _REG_EPS * np.eye(d)

        return weights, means, covariances

    def _em_run(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[dict, float]:
        weights, means, covariances = self._initialize_params(X, rng)
        ll_history: list[float] = []
        converged_iter = self.max_iter

        for iteration in range(self.max_iter):
            responsibilities, log_ll = self._e_step(X, weights, means, covariances)
            ll_history.append(log_ll)
            weights, means, covariances = self._m_step(X, responsibilities)

            if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) < self.tol:
                converged_iter = iteration + 1
                break
        else:
            converged_iter = len(ll_history)

        return {
            "weights": weights,
            "means": means,
            "covariances": covariances,
            "responsibilities": responsibilities,
            "ll_history": ll_history,
            "converged_iter": converged_iter,
        }, ll_history[-1]

    def fit(self, X: np.ndarray) -> "MyGMM":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X должен быть двумерным массивом (n_samples, n_features)")

        best_params = None
        best_log_likelihood = -np.inf
        base_seed = 0 if self.random_state is None else int(self.random_state)

        for init in range(self.n_init):
            rng = np.random.default_rng(base_seed + init)
            params, final_ll = self._em_run(X, rng)
            if self.verbose:
                print(
                    f"init {init + 1}/{self.n_init}: "
                    f"log L = {final_ll:.4f}, iter = {params['converged_iter']}"
                )
            min_weight = np.min(params["weights"])
            valid = min_weight >= 1.0 / (10 * self.n_components)
            if valid and final_ll > best_log_likelihood:
                best_log_likelihood = final_ll
                best_params = params
            elif best_params is None:
                best_log_likelihood = final_ll
                best_params = params

        self.weights_ = best_params["weights"]
        self.means_ = best_params["means"]
        self.covariances_ = best_params["covariances"]
        self.labels_ = np.argmax(best_params["responsibilities"], axis=1)
        self.log_likelihood_history_ = best_params["ll_history"]
        self.converged_iter_ = best_params["converged_iter"]
        self.n_iter_ = self.converged_iter_
        self.best_log_likelihood_ = best_log_likelihood
        return self

    def _check_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Модель не обучена: вызовите fit(X)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        responsibilities, _ = self._e_step(
            np.asarray(X, dtype=float), self.weights_, self.means_, self.covariances_
        )
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        responsibilities, _ = self._e_step(
            np.asarray(X, dtype=float), self.weights_, self.means_, self.covariances_
        )
        return responsibilities

    def score(self, X: np.ndarray) -> float:
        self._check_fitted()
        _, log_ll = self._e_step(
            np.asarray(X, dtype=float), self.weights_, self.means_, self.covariances_
        )
        return log_ll

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self._check_fitted()
        n, K = X.shape[0], self.n_components
        log_resp = np.empty((n, K))
        for k in range(K):
            log_resp[:, k] = np.log(self.weights_[k] + 1e-300) + _log_gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )
        return logsumexp(log_resp, axis=1)

    def compute_aic_bic(self, X: np.ndarray) -> tuple[float, float]:
        n, d = X.shape
        K = self.n_components
        n_params = K * d + K * d * (d + 1) // 2 + (K - 1)
        log_ll = self.score(X)
        aic = -2.0 * log_ll + 2.0 * n_params
        bic = -2.0 * log_ll + n_params * np.log(n)
        return float(aic), float(bic)
