| model | task | parameters | accuracy | precision | recall | f1 | roc_auc | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | classification | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 0.804469 | 0.826923 | 0.623188 | 0.710744 | 0.827141 | 0.0941204 | 0.00551315 |
| sklearn | classification | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 0.787709 | 0.762712 | 0.652174 | 0.703125 | 0.805797 | 0.128606 | 0.000627936 |
