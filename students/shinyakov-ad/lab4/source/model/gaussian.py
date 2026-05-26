import numpy as np


def regularize_cov(cov, reg_covar=1e-6):
    return cov + reg_covar * np.eye(cov.shape[0])


def log_multivariate_gaussian(X, mean, cov, reg_covar=1e-6):
    n_features = X.shape[1]
    cov = regularize_cov(cov, reg_covar)

    sign, logdet = np.linalg.slogdet(cov)
    inv_cov = np.linalg.pinv(cov)

    diff = X - mean
    transformed = np.einsum("ij,jk->ik", diff, inv_cov)
    exponent = -0.5 * np.sum(transformed * diff, axis=1)

    return -0.5 * (n_features * np.log(2 * np.pi) + logdet) + exponent
