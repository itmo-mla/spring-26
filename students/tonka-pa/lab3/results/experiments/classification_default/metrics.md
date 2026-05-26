| model | task | parameters | accuracy | precision | recall | f1 | roc_auc | fit_time | predict_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | classification | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 0.798883 | 0.866667 | 0.565217 | 0.684211 | 0.838208 | 0.0739239 | 0.00504279 |
| sklearn | classification | {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0, "random_state": 42} | 0.798883 | 0.789474 | 0.652174 | 0.714286 | 0.817918 | 0.0736022 | 0.000386125 |
