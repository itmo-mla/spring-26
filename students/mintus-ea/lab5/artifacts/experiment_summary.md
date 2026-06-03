# Experiment summary

Dataset: 20 Newsgroups subset
Documents: 360
Topics: 6
Terms: 180
Full matrix density: 0.0607
Train matrix density after holdout: 0.0494
Held-out entries: 733

## Parameters

`test_fraction=0.2`, `lsa_components=20`, `slim_l1=0.002`, `slim_l2=0.02`

## Metrics

| Model | Family | RMSE | NDCG@10 | Train time, s |
|---|---|---:|---:|---:|
| Custom SLIM | SLIM | 0.23404 | 0.09605 | 0.6915 |
| Sklearn ElasticNet SLIM | SLIM reference | 0.24290 | 0.10329 | 0.0845 |
| Custom LSA | Latent semantic | 0.23964 | 0.08869 | 0.0057 |
| Sklearn TruncatedSVD | Latent semantic reference | 0.23942 | 0.09058 | 0.0033 |
