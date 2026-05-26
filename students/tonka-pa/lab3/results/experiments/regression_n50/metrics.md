| model | task | parameters | mse | rmse | mae | r2 | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | regression | {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 504367 | 710.188 | 399.914 | 0.968272 | 1.67115 | 0.0137013 |
| sklearn | regression | {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 504216 | 710.081 | 399.871 | 0.968282 | 1.67971 | 0.00655274 |
