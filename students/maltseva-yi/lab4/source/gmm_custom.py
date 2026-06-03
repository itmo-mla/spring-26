import numpy as np
from scipy.special import logsumexp

class GaussianMixtureEM:
    # Gaussian Mixture Model с обучением через EM-алгоритм.
    def __init__(self, n_components=3, max_iter=200, tol=1e-4, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_ = []

    def _initialize_parameters(self, X):
        # Инициализация параметров: веса, средние, ковариации.
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # средние – случайные точки из выборки
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()

        # ковариации – глобальная ковариация выборки + регуляризация
        global_cov = np.cov(X.T) + self.reg_covar * np.eye(n_features)
        self.covariances_ = np.array([global_cov.copy() for _ in range(self.n_components)])

        # веса – равномерные
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

    def _log_multivariate_normal_density(self, X, mean, cov):
        # Логарифм плотности многомерного нормального распределения.
        n_features = X.shape[1]
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov_reg = cov + self.reg_covar * np.eye(n_features)
            L = np.linalg.cholesky(cov_reg)

        X_centered = X - mean
        solve_L = np.linalg.solve(L, X_centered.T).T
        log_det = 2 * np.sum(np.log(np.diag(L)))
        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det + np.sum(solve_L**2, axis=1))
        return log_prob

    def _e_step(self, X):
        # E-шаг: вычисление апостериорных вероятностей и логарифма правдоподобия.
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_components))
        for j in range(self.n_components):
            try:
                log_prob[:, j] = self._log_multivariate_normal_density(X, self.means_[j], self.covariances_[j])
            except:
                cov_reg = self.covariances_[j] + self.reg_covar * np.eye(X.shape[1])
                log_prob[:, j] = self._log_multivariate_normal_density(X, self.means_[j], cov_reg)

        log_weights = np.log(self.weights_)
        log_joint = log_prob + log_weights
        log_likelihood = logsumexp(log_joint, axis=1).sum()
        log_posterior = log_joint - logsumexp(log_joint, axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        return posterior, log_likelihood

    def _m_step(self, X, posterior):
        # M-шаг: обновление весов, средних и ковариаций.
        n_samples, n_features = X.shape
        Nk = posterior.sum(axis=0)
        self.weights_ = Nk / n_samples
        self.means_ = (posterior.T @ X) / Nk[:, np.newaxis]

        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for j in range(self.n_components):
            X_centered = X - self.means_[j]
            weighted_cov = (posterior[:, j][:, np.newaxis] * X_centered).T @ X_centered
            cov_j = weighted_cov / Nk[j]
            cov_j.flat[::n_features+1] += self.reg_covar
            self.covariances_[j] = cov_j

    def fit(self, X):
        # Обучение модели EM-алгоритмом.
        X = np.asarray(X)
        self._initialize_parameters(X)
        prev_log_likelihood = -np.inf
        self.log_likelihood_ = []

        for iteration in range(self.max_iter):
            posterior, log_likelihood = self._e_step(X)
            self.log_likelihood_.append(log_likelihood)

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break

            self._m_step(X, posterior)
            prev_log_likelihood = log_likelihood

        return self

    def score(self, X):
        # Логарифм правдоподобия выборки (сумма по объектам).
        X = np.asarray(X)
        log_prob = np.zeros((X.shape[0], self.n_components))
        for j in range(self.n_components):
            try:
                log_prob[:, j] = self._log_multivariate_normal_density(X, self.means_[j], self.covariances_[j])
            except:
                cov_reg = self.covariances_[j] + self.reg_covar * np.eye(X.shape[1])
                log_prob[:, j] = self._log_multivariate_normal_density(X, self.means_[j], cov_reg)

        log_weights = np.log(self.weights_)
        log_joint = log_prob + log_weights
        total_log_likelihood = logsumexp(log_joint, axis=1).sum()
        return total_log_likelihood

    def predict_proba(self, X):
        # Апостериорные вероятности принадлежности компонентам.
        X = np.asarray(X)
        log_prob = np.zeros((X.shape[0], self.n_components))
        for j in range(self.n_components):
            try:
                log_prob[:, j] = self._log_multivariate_normal_density(X, self.means_[j], self.covariances_[j])
            except:
                cov_reg = self.covariances_[j] + self.reg_covar * np.eye(X.shape[1])
                log_prob[:, j] = self._log_multivariate_normal_density(X, self.means_[j], cov_reg)

        log_weights = np.log(self.weights_)
        log_joint = log_prob + log_weights
        log_posterior = log_joint - logsumexp(log_joint, axis=1, keepdims=True)
        return np.exp(log_posterior)

    def predict(self, X):
        # Предсказание наиболее вероятной компоненты.
        return np.argmax(self.predict_proba(X), axis=1)