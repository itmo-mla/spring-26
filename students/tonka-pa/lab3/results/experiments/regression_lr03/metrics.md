| model | task | parameters | mse | rmse | mae | r2 | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | regression | {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 318173 | 564.069 | 313.286 | 0.979985 | 3.45954 | 0.0191289 |
| sklearn | regression | {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 319334 | 565.097 | 313.501 | 0.979912 | 3.38173 | 0.00832501 |
