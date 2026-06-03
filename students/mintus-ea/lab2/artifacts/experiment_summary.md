# Experiment summary

Dataset: sklearn Optical Recognition of Handwritten Digits
Samples: 1797
Features: 64 (8x8 pixels)
Classes: 10

## Best custom RF parameters

`max_depth=None`, `max_features=sqrt`, `min_samples_leaf=2`, `n_estimators=80`
Best OOB accuracy during grid search: `0.9614`
Grid search time: `8.3043` seconds

## Metrics

| Model | Accuracy | Precision macro | Recall macro | F1 macro | OOB score | Train time, s |
|---|---:|---:|---:|---:|---:|---:|
| Custom Random Forest | 0.9750 | 0.9754 | 0.9747 | 0.9746 | 0.9673 | 0.1742 |
| Sklearn RandomForest | 0.9611 | 0.9621 | 0.9606 | 0.9605 | 0.9701 | 0.0943 |

## Top OOB^j importances

| Feature | OOB accuracy decrease | Std |
|---|---:|---:|
| pixel_3_2 | 0.00858 | 0.00269 |
| pixel_2_5 | 0.00812 | 0.00066 |
| pixel_4_6 | 0.00696 | 0.00098 |
| pixel_5_2 | 0.00580 | 0.00200 |
| pixel_5_3 | 0.00580 | 0.00143 |
| pixel_6_2 | 0.00464 | 0.00066 |
| pixel_2_2 | 0.00418 | 0.00098 |
| pixel_6_3 | 0.00371 | 0.00183 |
