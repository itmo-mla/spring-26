import numpy as np


def regularize_cov(cov, reg_covar=1e-6):
    return cov + reg_covar * np.eye(cov.shape[0])

# Функция вычисления многомерной гауссовской плотности вероятности
def multivariate_gaussian(X, mean, cov):
    n_features = X.shape[1]
    cov = regularize_cov(cov)

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** n_features) * det_cov)

    diff = X - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

    return norm_const * np.exp(exponent)

# Логарифм многомерной гауссовской плотности вероятности
def log_multivariate_gaussian(X, mean, cov):
    n_features = X.shape[1]
    cov = regularize_cov(cov)

    sign, logdet = np.linalg.slogdet(cov)
    inv_cov = np.linalg.inv(cov)

    diff = X - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

    return -0.5 * (n_features * np.log(2 * np.pi) + logdet) + exponent