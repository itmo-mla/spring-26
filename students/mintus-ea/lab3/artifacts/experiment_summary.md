# Experiment summary

Dataset: sklearn Breast Cancer Wisconsin
Samples: 569
Features: 30
Positive class: malignant

## Model parameters

`n_estimators=140`, `learning_rate=0.08`, `max_depth=2`, `subsample=1.0`, `min_samples_leaf=4`

## Cross-validation

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Log loss | Train time, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom Gradient Boosting | 0.9495 ± 0.0164 | 0.9651 ± 0.0407 | 0.9000 ± 0.0440 | 0.9299 ± 0.0231 | 0.9843 ± 0.0075 | 0.1959 ± 0.0261 | 1.8717 |
| Sklearn GradientBoosting | 0.9714 ± 0.0112 | 0.9764 ± 0.0216 | 0.9471 ± 0.0288 | 0.9611 ± 0.0156 | 0.9916 ± 0.0052 | 0.0923 ± 0.0333 | 0.2309 |

## Hold-out test

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Log loss | Train time, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom Gradient Boosting | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9937 | 0.1922 | 2.3072 |
| Sklearn GradientBoosting | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9911 | 0.1046 | 0.2807 |
