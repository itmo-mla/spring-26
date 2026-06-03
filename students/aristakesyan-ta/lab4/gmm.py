import numpy as np


def _logsumexp(values, axis=None, keepdims=False):
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    result = max_values + np.log(np.sum(shifted, axis=axis, keepdims=True))

    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


class GaussianMixtureModel:
    def __init__(
        self,
        n_components=3,
        max_iter=100,
        tol=1e-4,
        reg_covar=1e-6,
        init_params="kmeans",
        random_state=None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_params = init_params
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf
        self.lower_bound_history_ = []

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._validate_input(X)
        self._initialize_parameters(X)

        previous_lower_bound = -np.inf
        self.lower_bound_history_ = []
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            log_prob_norm, responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            current_lower_bound = float(np.mean(log_prob_norm))
            self.lower_bound_history_.append(current_lower_bound)
            change = current_lower_bound - previous_lower_bound

            self.n_iter_ = iteration
            self.lower_bound_ = current_lower_bound

            if abs(change) < self.tol:
                self.converged_ = True
                break

            previous_lower_bound = current_lower_bound

        return self

    def predict_proba(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        _, responsibilities = self._e_step(X)
        return responsibilities

    def predict(self, X):
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)

    def score_samples(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        return _logsumexp(weighted_log_prob, axis=1)

    def score(self, X):
        return float(np.mean(self.score_samples(X)))

    def get_params(self, deep=True):
        return {
            "n_components": self.n_components,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "init_params": self.init_params,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        indices = rng.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[indices].copy()
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

        if self.init_params == "kmeans":
            self.means_ = self._run_kmeans_initialization(X, rng)
        elif self.init_params != "random":
            raise ValueError("init_params must be 'kmeans' or 'random'.")

        global_covariance = np.cov(X, rowvar=False)
        if global_covariance.ndim == 0:
            global_covariance = np.array([[float(global_covariance)]])
        global_covariance += self.reg_covar * np.eye(n_features)
        self.covariances_ = np.array(
            [global_covariance.copy() for _ in range(self.n_components)]
        )

    def _run_kmeans_initialization(self, X, rng, n_iter=10):
        means = self.means_.copy()

        for _ in range(n_iter):
            distances = np.linalg.norm(X[:, np.newaxis, :] - means[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            for component_idx in range(self.n_components):
                cluster_points = X[labels == component_idx]
                if len(cluster_points) == 0:
                    means[component_idx] = X[rng.integers(0, X.shape[0])]
                else:
                    means[component_idx] = np.mean(cluster_points, axis=0)

        return means

    def _e_step(self, X):
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = _logsumexp(weighted_log_prob, axis=1)
        log_responsibilities = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, np.exp(log_responsibilities)

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        component_weights = responsibilities.sum(axis=0) + 10 * np.finfo(float).eps

        self.weights_ = component_weights / n_samples
        self.means_ = (responsibilities.T @ X) / component_weights[:, np.newaxis]

        covariances = np.empty((self.n_components, n_features, n_features))
        for component_idx in range(self.n_components):
            diff = X - self.means_[component_idx]
            weighted_diff = responsibilities[:, component_idx][:, np.newaxis] * diff
            covariance = (weighted_diff.T @ diff) / component_weights[component_idx]
            covariance += self.reg_covar * np.eye(n_features)
            covariances[component_idx] = covariance

        self.covariances_ = covariances

    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_gaussian_prob(X) + np.log(self.weights_)

    def _estimate_log_gaussian_prob(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))

        for component_idx in range(self.n_components):
            mean = self.means_[component_idx]
            covariance = self.covariances_[component_idx]

            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0:
                covariance = covariance + self.reg_covar * np.eye(n_features)
                sign, log_det = np.linalg.slogdet(covariance)

            diff = X - mean
            solved = np.linalg.solve(covariance, diff.T).T
            mahalanobis = np.sum(diff * solved, axis=1)

            log_prob[:, component_idx] = -0.5 * (
                n_features * np.log(2 * np.pi) + log_det + mahalanobis
            )

        return log_prob

    def _validate_input(self, X):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] < self.n_components:
            raise ValueError("n_components must not exceed number of samples.")
        if self.n_components < 1:
            raise ValueError("n_components must be positive.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")
        if self.reg_covar < 0:
            raise ValueError("reg_covar must be non-negative.")
        if self.init_params not in {"kmeans", "random"}:
            raise ValueError("init_params must be 'kmeans' or 'random'.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")

    def _check_is_fitted(self):
        if self.weights_ is None or self.means_ is None or self.covariances_ is None:
            raise ValueError("Model is not fitted yet.")
