| model | task | parameters | accuracy | precision | recall | f1 | roc_auc | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | classification | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 0.8, "random_state": 42} | 0.798883 | 0.851064 | 0.57971 | 0.689655 | 0.834914 | 0.0740501 | 0.00500835 |
| sklearn | classification | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 0.8, "random_state": 42} | 0.810056 | 0.79661 | 0.681159 | 0.734375 | 0.81693 | 0.0754266 | 0.000438305 |
