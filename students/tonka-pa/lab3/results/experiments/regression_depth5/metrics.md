| model | task | parameters | mse | rmse | mae | r2 | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | regression | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 284711 | 533.583 | 279.885 | 0.98209 | 5.47731 | 0.0297043 |
| sklearn | regression | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 284726 | 533.597 | 279.858 | 0.982089 | 5.5237 | 0.0185118 |
