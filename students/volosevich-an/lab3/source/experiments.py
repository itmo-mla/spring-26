# %%
from utils import load_data, cross_validate
from grad_boost import GradientBoostingClassifierCustom

from sklearn.ensemble import GradientBoostingClassifier

# %%
X, y = load_data("../data/train.csv")

# %%
custom_results = cross_validate(
    GradientBoostingClassifierCustom,
    X,
    y,
    model_params={
        "n_estimators": 25,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": 42
    },
    n_splits=5
)

print("Custom Gradient Boosting")
print(custom_results)

# %%
sklearn_results = cross_validate(
    GradientBoostingClassifier,
    X,
    y,
    model_params={
        "n_estimators": 25,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": 42
    },
    n_splits=5
)

print("\nSklearn Gradient Boosting")
print(sklearn_results)

# %%
print("\nComparison:")
print(f"Accuracy difference: {sklearn_results['mean_accuracy'] - custom_results['mean_accuracy']:.4f}")
print(f"Training time difference: {custom_results['mean_train_time'] - sklearn_results['mean_train_time']:.4f} sec")


