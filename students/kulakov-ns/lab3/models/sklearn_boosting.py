import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from utils.training import fit_grid_search



def get_sklearn_boosting(X_train: pd.DataFrame, y_train: pd.Series):
    estimator = GradientBoostingClassifier(
        loss="log_loss",
        min_samples_split=2,
        random_state=42,
    )
    return fit_grid_search(estimator, X_train, y_train)
