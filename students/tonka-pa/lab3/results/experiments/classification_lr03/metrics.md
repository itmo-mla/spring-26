| model | task | parameters | accuracy | precision | recall | f1 | roc_auc | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | classification | {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 0.804469 | 0.84 | 0.608696 | 0.705882 | 0.820751 | 0.0754265 | 0.00502226 |
| sklearn | classification | {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 0.815642 | 0.810345 | 0.681159 | 0.740157 | 0.793874 | 0.0783303 | 0.000400733 |
