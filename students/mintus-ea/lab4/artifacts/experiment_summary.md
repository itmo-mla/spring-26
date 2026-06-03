# Experiment summary

Dataset: sklearn Wine recognition
Samples: 178
Features: 13
Components: 3

## Parameters

`n_components=3`, `covariance_type=full`, `max_iter=200`, `tol=0.0001`, `reg_covar=1e-05`

## Metrics

| Model | Train avg LL | Test avg LL | Test total LL | BIC | AIC | Clustering acc | ARI | NMI | Iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom GMM EM | -11.4643 | -15.1707 | -682.68 | 2560.66 | 1993.36 | 0.9556 | 0.8575 | 0.8728 | 22 |
| Sklearn GaussianMixture | -11.4643 | -15.1707 | -682.68 | 2560.66 | 1993.36 | 0.9556 | 0.8575 | 0.8728 | 22 |

## Training time

- Custom GMM EM: `0.0146` seconds
- Sklearn GaussianMixture: `0.0441` seconds
