# Experiment summary

Sample size: 6000
Pruned internal nodes: 30

## Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Custom ID3 before pruning | 0.8567 | 0.7577 | 0.5952 | 0.6667 | 0.8872 |
| Custom ID3 after pruning | 0.8608 | 0.7933 | 0.5709 | 0.6640 | 0.8619 |
| Sklearn DecisionTree | 0.8592 | 0.7703 | 0.5917 | 0.6693 | 0.8848 |

## Tree complexity

| State | Depth | Nodes | Leaves |
|---|---:|---:|---:|
| Before pruning | 8 | 81 | 41 |
| After pruning | 6 | 21 | 11 |
