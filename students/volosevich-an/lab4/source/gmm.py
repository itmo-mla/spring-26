import numpy as np
from gaussian import log_multivariate_gaussian
from init_param import random_init


class EMGMM:
    
    def __init__(
        self,
        n_components=3,
        max_iter=100,
        tol=1e-4,
        reg_covar=1e-6,
        random_state=2
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_ = None      # w_j — априорные вероятности компонент 
        self.means_ = None        # μ_j — центры компонент
        self.covariances_ = None  # Σ_j — ковариационные матрицы компонент
        self.log_likelihood_history_ = []
        self.responsibilities_history_ = []  

    def _e_step(self, X):
        n_samples = X.shape[0]
        
        # Вычисляем log(w_j * p(x_i | θ_j))
        log_resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            log_resp[:, k] = (
                np.log(self.weights_[k] + 1e-12)
                + log_multivariate_gaussian(
                    X,
                    self.means_[k],
                    self.covariances_[k]
                )
            )
        
        # Логарифмическое суммирование для численной устойчивости
        # log Σ exp(log_resp) = max_log + log Σ exp(log_resp - max_log)
        max_log = np.max(log_resp, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(
            np.sum(np.exp(log_resp - max_log), axis=1, keepdims=True)
        )
        
        # Логарифм правдоподобия выборки: L = Σ_i ln p(x_i) 
        log_likelihood = np.sum(log_sum_exp)
        
        # Апостериорные вероятности (нормированные) — выполняется Σ_j g_ij = 1 
        responsibilities = np.exp(log_resp - log_sum_exp)
        
        return responsibilities, log_likelihood
    
    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        
        # Nk = Σ_i g_ij = ℓ * w_j
        Nk = responsibilities.sum(axis=0)
        
        # Обновление весов смеси w_j 
        self.weights_ = Nk / n_samples
        
        # Обновление центров компонент μ_j 
        self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]
        
        # Обновление ковариационных матриц Σ_j 
        new_covariances = []
        
        for k in range(self.n_components):
            # Разности x_i - μ_j
            diff = X - self.means_[k]
            
            # Взвешенная ковариационная матрица
            # Σ_j = (1/(ℓ w_j)) * Σ_i g_ij * (x_i - μ_j)(x_i - μ_j)^T
            weighted_diff = responsibilities[:, k][:, np.newaxis] * diff
            cov = weighted_diff.T @ diff / Nk[k]
            
            # Регуляризация: Σ + τI 
            # Проблема мультиколлинеарности: при ℓ < n матрица вырождена,
            # регуляризация увеличивает собственные значения на τ
            cov += self.reg_covar * np.eye(n_features)
            
            new_covariances.append(cov)
        
        self.covariances_ = np.array(new_covariances)
    
    def _check_convergence(self, old_responsibilities, new_responsibilities):
        """
        Проверка сходимости EM-алгоритма по g_ij (апостериорным вероятностям).
        
        По лекции (стр. 11): "Пока w_j, θ_j и/или g_ij не сошлись"
        """
        if old_responsibilities is None:
            return False
        
        diff = np.abs(new_responsibilities - old_responsibilities)
        return np.max(diff) < self.tol
    
    def fit(self, X):
        # Инициализация параметров (стр. 11: w_j = 1/k, μ_j и Σ_j через random_init)
        self.weights_, self.means_, self.covariances_ = random_init(
            X,
            self.n_components,
            self.random_state
        )
        
        prev_responsibilities = None
        self.log_likelihood_history_ = []
        self.responsibilities_history_ = []
        
        for iteration in range(self.max_iter):
            # E-шаг: вычисление апостериорных вероятностей (стр. 8, 11)
            responsibilities, log_likelihood = self._e_step(X)
            
            # M-шаг: обновление параметров 
            self._m_step(X, responsibilities)
            
            # Сохраняем историю для анализа
            self.log_likelihood_history_.append(log_likelihood)
            self.responsibilities_history_.append(responsibilities.copy())
            
            # Проверка сходимости по g_ij 
            if self._check_convergence(prev_responsibilities, responsibilities):
                break
            
            prev_responsibilities = responsibilities
        
        return self
    
    def predict_proba(self, X):

        responsibilities, _ = self._e_step(X)
        return responsibilities
    
    def predict(self, X):
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)
    