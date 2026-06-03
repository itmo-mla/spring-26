import pandas as pd
from sklearn.mixture import GaussianMixture

from utils.training import fit_grid_search



def get_sklearn_gmm(X_train: pd.DataFrame):
    estimator = GaussianMixture(
        covariance_type="full",
        max_iter=200,
        n_init=5,
        random_state=42,
    )
    return fit_grid_search(estimator, X_train)
