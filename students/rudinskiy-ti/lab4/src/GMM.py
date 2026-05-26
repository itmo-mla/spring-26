import numpy as np

def multivariate_gaussian_pdf(X, mean, cov):
    n_features = X.shape[1]
    cov = cov + np.eye(n_features) * 1e-6
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** n_features) * det_cov)
    diff = X - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
    return norm_const * np.exp(exponent)

class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.responsibilities_ = None
        self.log_likelihood_history_ = []

    def _initialize_parameters(self, X):
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        random_indices = np.random.choice(
            n_samples, 
            self.n_components, 
            replace=False
        )

        self.means_ = X[random_indices]

        self.covariances_ = np.array([
            np.cov(X.T) + np.eye(n_features) * 1e-6
            for _ in range(self.n_components)
        ])

        self.weights_ = np.ones(self.n_components) / self.n_components

    def _e_step(self, X):
        n_samples = X.shape[0]

        responsibilities = np.zeros((n_samples, self.n_components))

        for j in range(self.n_components):
            responsibilities[:, j] = (
                self.weights_[j] *
                multivariate_gaussian_pdf(
                    X, 
                    self.means_[j], 
                    self.covariances_[j]
                )
            )

        denominator = responsibilities.sum(axis=1, keepdims=True)
        denominator = np.maximum(denominator, 1e-12)

        responsibilities = responsibilities / denominator

        self.responsibilities_ = responsibilities

    def _m_step(self, X):
        n_samples, n_features = X.shape

        N_k = self.responsibilities_.sum(axis=0)

        self.weights_ = N_k / n_samples

        self.means_ = np.zeros((self.n_components, n_features))
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))

        for j in range(self.n_components):
            gamma_j = self.responsibilities_[:, j]

            self.means_[j] = np.sum(gamma_j[:, np.newaxis] * X, axis=0) / N_k[j]

            diff = X - self.means_[j]

            self.covariances_[j] = (
                (gamma_j[:, np.newaxis] * diff).T @ diff
            ) / N_k[j]

            self.covariances_[j] += np.eye(n_features) * 1e-6

    def score_samples(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_components))

        for j in range(self.n_components):
            probs[:, j] = (
                self.weights_[j] *
                multivariate_gaussian_pdf(
                    X, 
                    self.means_[j], 
                    self.covariances_[j]
                )
            )

        total_probs = probs.sum(axis=1)
        total_probs = np.maximum(total_probs, 1e-12)

        return np.log(total_probs)

    def score(self, X):
        return np.mean(self.score_samples(X))

    def log_likelihood(self, X):
        return np.sum(self.score_samples(X))

    def fit(self, X):
        self._initialize_parameters(X)

        prev_log_likelihood = None

        for iteration in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)

            current_log_likelihood = self.log_likelihood(X)
            self.log_likelihood_history_.append(current_log_likelihood)

            if prev_log_likelihood is not None:
                diff = abs(current_log_likelihood - prev_log_likelihood)

                if diff < self.tol:
                    print(f"Сошлось! {iteration}")
                    break

            prev_log_likelihood = current_log_likelihood

        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]

        responsibilities = np.zeros((n_samples, self.n_components))

        for j in range(self.n_components):
            responsibilities[:, j] = (
                self.weights_[j] *
                multivariate_gaussian_pdf(
                    X, 
                    self.means_[j], 
                    self.covariances_[j]
                )
            )

        denominator = responsibilities.sum(axis=1, keepdims=True)
        denominator = np.maximum(denominator, 1e-12)

        return responsibilities / denominator

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)