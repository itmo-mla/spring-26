import numpy as np
from scipy.special import logsumexp


def multivariate_gaussian_log_pdf(X, mean, cov):
    n_features = X.shape[1]
    diff = X - mean
    try:
        L = np.linalg.cholesky(cov)
        y = np.linalg.solve(L, diff.T)
        mahal = np.sum(y ** 2, axis=0)
    except np.linalg.LinAlgError:
        mahal = np.sum(diff @ np.linalg.pinv(cov) * diff, axis=1)

    sign, log_det = np.linalg.slogdet(cov)
    log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)
    return log_prob