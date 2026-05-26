| model | task | parameters | mse | rmse | mae | r2 | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | regression | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 0.8, "random_state": 42} | 369431 | 607.808 | 338.8 | 0.976761 | 2.91999 | 0.0219263 |
| sklearn | regression | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 0.8, "random_state": 42} | 375835 | 613.054 | 345.843 | 0.976358 | 2.80559 | 0.0108694 |
