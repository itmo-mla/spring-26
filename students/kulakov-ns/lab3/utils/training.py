from sklearn.model_selection import GridSearchCV, StratifiedKFold


PARAM_GRID = {
    "n_estimators": [20, 40, 60],
    "learning_rate": [0.05, 0.1],
    "max_depth": [1, 2],
    "min_samples_leaf": [1],
}


def get_cv(random_state: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)



def fit_grid_search(estimator, X_train, y_train, scoring: str = "accuracy") -> GridSearchCV:
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=PARAM_GRID,
        scoring=scoring,
        cv=get_cv(),
        refit=True,
        n_jobs=1,
    )
    return grid_search.fit(X_train, y_train)
