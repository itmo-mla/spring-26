| model | task | parameters | accuracy | precision | recall | f1 | roc_auc | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | classification | {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 0.798883 | 0.866667 | 0.565217 | 0.684211 | 0.83834 | 0.0445799 | 0.00365151 |
| sklearn | classification | {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 0.810056 | 0.843137 | 0.623188 | 0.716667 | 0.82556 | 0.041158 | 0.000303719 |
