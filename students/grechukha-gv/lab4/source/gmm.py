import numpy as np


class GaussianMixtureEM:
    """Gaussian Mixture Model with full covariance matrices trained by EM"""

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        n_init: int = 5,
        init_params: str = "kmeans++",
        random_state: int = 42,
    ) -> None:
        if init_params not in {"kmeans++", "random"}:
            raise ValueError("init_params must be 'kmeans++' or 'random'")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf
        self.log_likelihood_history_: list[float] = []

    def fit(self, x: np.ndarray) -> "GaussianMixtureEM":
        x = self._validate_x(x)
        if x.shape[0] < self.n_components:
            raise ValueError("n_components must not exceed number of samples")

        rng = np.random.default_rng(self.random_state)
        best_state: dict[str, np.ndarray | bool | float | int | list[float]] | None = None

        for _ in range(self.n_init):
            self._initialize_parameters(x, rng)
            previous_lower_bound = -np.inf
            history: list[float] = []
            converged = False
            n_iter = self.max_iter

            for iteration in range(1, self.max_iter + 1):
                log_responsibilities, log_likelihood = self._e_step(x)
                self._m_step(x, log_responsibilities)

                lower_bound = log_likelihood / x.shape[0]
                history.append(lower_bound)
                change = lower_bound - previous_lower_bound
                if abs(change) < self.tol:
                    converged = True
                    n_iter = iteration
                    break
                previous_lower_bound = lower_bound

            if history and history[-1] > (best_state["lower_bound"] if best_state else -np.inf):
                best_state = {
                    "weights": self.weights_.copy(),
                    "means": self.means_.copy(),
                    "covariances": self.covariances_.copy(),
                    "converged": converged,
                    "n_iter": n_iter,
                    "lower_bound": history[-1],
                    "history": history,
                }

        if best_state is None:
            raise RuntimeError("failed to fit GMM")

        self.weights_ = best_state["weights"]
        self.means_ = best_state["means"]
        self.covariances_ = best_state["covariances"]
        self.converged_ = bool(best_state["converged"])
        self.n_iter_ = int(best_state["n_iter"])
        self.lower_bound_ = float(best_state["lower_bound"])
        self.log_likelihood_history_ = list(best_state["history"])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        log_responsibilities, _ = self._e_step(self._validate_x(x))
        return np.exp(log_responsibilities)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        weighted_log_prob = self._estimate_weighted_log_prob(self._validate_x(x))
        return np.logaddexp.reduce(weighted_log_prob, axis=1)

    def score(self, x: np.ndarray) -> float:
        return float(np.mean(self.score_samples(x)))

    def aic(self, x: np.ndarray) -> float:
        return -2.0 * np.sum(self.score_samples(x)) + 2.0 * self._n_parameters(x)

    def bic(self, x: np.ndarray) -> float:
        n_samples = self._validate_x(x).shape[0]
        return -2.0 * np.sum(self.score_samples(x)) + self._n_parameters(x) * np.log(n_samples)

    def _initialize_parameters(self, x: np.ndarray, rng: np.random.Generator) -> None:
        n_samples, n_features = x.shape

        if self.init_params == "random":
            indices = rng.choice(n_samples, self.n_components, replace=False)
            self.means_ = x[indices].copy()
            self.covariances_ = np.array(
                [np.eye(n_features) + self.reg_covar * np.eye(n_features) for _ in range(self.n_components)]
            )
            self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
            return

        centers = self._run_kmeans(x, self._kmeans_plus_plus(x, rng))
        distances = np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)

        global_covariance = np.cov(x, rowvar=False) + self.reg_covar * np.eye(n_features)
        weights = np.empty(self.n_components, dtype=float)
        means = np.empty((self.n_components, n_features), dtype=float)
        covariances = np.empty((self.n_components, n_features, n_features), dtype=float)

        for component in range(self.n_components):
            mask = assignments == component
            if np.sum(mask) <= 1:
                means[component] = centers[component]
                covariances[component] = global_covariance.copy()
                weights[component] = 1.0 / self.n_components
            else:
                cluster = x[mask]
                means[component] = np.mean(cluster, axis=0)
                covariances[component] = np.cov(cluster, rowvar=False) + self.reg_covar * np.eye(n_features)
                weights[component] = cluster.shape[0] / n_samples

        self.weights_ = weights / np.sum(weights)
        self.means_ = means
        self.covariances_ = covariances

    def _kmeans_plus_plus(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples = x.shape[0]
        centers = [x[rng.integers(n_samples)].copy()]

        for _ in range(1, self.n_components):
            distances = np.min(np.sum((x[:, np.newaxis, :] - np.array(centers)[np.newaxis, :, :]) ** 2, axis=2), axis=1)
            if np.allclose(distances.sum(), 0.0):
                centers.append(x[rng.integers(n_samples)].copy())
                continue
            probabilities = distances / distances.sum()
            centers.append(x[rng.choice(n_samples, p=probabilities)].copy())

        return np.array(centers)

    def _run_kmeans(self, x: np.ndarray, centers: np.ndarray, max_iter: int = 30) -> np.ndarray:
        for _ in range(max_iter):
            distances = np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
            assignments = np.argmin(distances, axis=1)
            new_centers = centers.copy()

            for component in range(self.n_components):
                mask = assignments == component
                if np.any(mask):
                    new_centers[component] = np.mean(x[mask], axis=0)

            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        return centers

    def _e_step(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        weighted_log_prob = self._estimate_weighted_log_prob(x)
        log_prob_norm = np.logaddexp.reduce(weighted_log_prob, axis=1)
        log_responsibilities = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_responsibilities, float(np.sum(log_prob_norm))

    def _m_step(self, x: np.ndarray, log_responsibilities: np.ndarray) -> None:
        responsibilities = np.exp(log_responsibilities)
        n_samples, n_features = x.shape
        component_sizes = responsibilities.sum(axis=0) + 10 * np.finfo(float).eps

        self.weights_ = component_sizes / n_samples
        self.means_ = (responsibilities.T @ x) / component_sizes[:, np.newaxis]

        covariances = np.empty((self.n_components, n_features, n_features), dtype=float)
        for component in range(self.n_components):
            centered = x - self.means_[component]
            covariances[component] = (
                responsibilities[:, component][:, np.newaxis] * centered
            ).T @ centered / component_sizes[component]
            covariances[component].flat[:: n_features + 1] += self.reg_covar
        self.covariances_ = covariances

    def _estimate_weighted_log_prob(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.means_ is None or self.covariances_ is None:
            raise ValueError("model is not fitted")

        log_prob = self._estimate_log_gaussian_prob(x)
        return log_prob + np.log(self.weights_ + np.finfo(float).eps)

    def _estimate_log_gaussian_prob(self, x: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.covariances_ is None:
            raise ValueError("model is not fitted")

        n_samples, n_features = x.shape
        log_prob = np.empty((n_samples, self.n_components), dtype=float)

        for component, (mean, covariance) in enumerate(zip(self.means_, self.covariances_, strict=True)):
            stable_covariance = self._make_positive_definite(covariance)
            sign, log_det = np.linalg.slogdet(stable_covariance)
            if sign <= 0:
                raise np.linalg.LinAlgError("covariance matrix must be positive definite")
            centered = x - mean
            solved = np.linalg.solve(stable_covariance, centered.T).T
            mahalanobis = np.sum(centered * solved, axis=1)
            log_prob[:, component] = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + mahalanobis)

        return log_prob

    def _make_positive_definite(self, covariance: np.ndarray) -> np.ndarray:
        n_features = covariance.shape[0]
        for multiplier in (0.0, 1.0, 10.0, 100.0, 1000.0):
            candidate = covariance + multiplier * self.reg_covar * np.eye(n_features)
            try:
                np.linalg.cholesky(candidate)
            except np.linalg.LinAlgError:
                continue
            return candidate
        raise np.linalg.LinAlgError("could not regularize covariance matrix")

    def _n_parameters(self, x: np.ndarray) -> int:
        _, n_features = self._validate_x(x).shape
        covariance_parameters = self.n_components * n_features * (n_features + 1) // 2
        mean_parameters = self.n_components * n_features
        weight_parameters = self.n_components - 1
        return covariance_parameters + mean_parameters + weight_parameters

    @staticmethod
    def _validate_x(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain finite values")
        return x
