| model | task | parameters | mse | rmse | mae | r2 | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | regression | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 371092 | 609.173 | 336.545 | 0.976656 | 3.29182 | 0.0212796 |
| sklearn | regression | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 371046 | 609.136 | 336.522 | 0.976659 | 3.29621 | 0.0111161 |
