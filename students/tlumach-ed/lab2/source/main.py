from dataset import load_dataset
from ensemble.random_forest import RandomForest
from metrics import classification_metrics

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import numpy as np
import time


def plot_feature_importance(importances, feature_names, top_n=20):
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha="right")

    plt.title(f"Top-{top_n} Feature Importance (OOB)")
    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.tight_layout()
    plt.show()



def plot_oob_curve(X, y):
    oob_errors = []
    estimators_range = range(10, 150, 10)

    for n in estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n,
            oob_score=True,
            bootstrap=True,
            max_features="sqrt"
        )
        rf.fit(X, y)
        oob_errors.append(1 - rf.oob_score_)

    plt.figure(figsize=(8, 5))
    plt.plot(estimators_range, oob_errors)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error")
    plt.title("OOB Error vs Number of Trees")
    plt.show()



def custom_grid_search(X, y):
    best_score = 0
    best_params = None

    for n_trees in [50, 100]:
        for max_depth in [6, 10, 12]:
            for min_samples_split in [5, 10]:
                rf = RandomForest(
                    n_trees=n_trees,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )

                rf.fit(X, y)
                score = rf.oob_score(X, y)

                print(f"n_trees={n_trees}, depth={max_depth}, min_split={min_samples_split} → OOB={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = (n_trees, max_depth, min_samples_split)

    return best_params, best_score


X_train, X_test, y_train, y_test, feature_names = load_dataset()

# print("\n=== Custom GridSearch (OOB) ===")
# best_params, best_oob = custom_grid_search(X_train, y_train)
# print("\nBest params:", best_params)
# print("Best OOB score:", best_oob)

best_params = (100, 12, 5) # были найдены парой строк выше

#Custom Random Forest
start = time.time()
rf = RandomForest(
    n_trees=best_params[0],
    max_depth=best_params[1],
    min_samples_split=best_params[2]
)

rf.fit(X_train, y_train)
train_time = time.time() - start


pred = rf.predict(X_test)
metrics = classification_metrics(y_test, pred)

print("\n=== Final Model ===")
print(metrics)
print("OOB score:", rf.oob_score(X_train, y_train))
print("Train time:", train_time)



# Feature Importance
importances = rf.feature_importance(X_train, y_train)

print("\nFeature importance (first 10):", importances[:10])
plot_feature_importance(importances, feature_names, top_n=10)


# Sklearn Random Forest
print("\n=== Sklearn Random Forest ===")
start = time.time()
sk_rf = RandomForestClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1]
)
sk_rf.fit(X_train, y_train)
train_time_sk = time.time() - start
pred_sk = sk_rf.predict(X_test)
metrics_sk = classification_metrics(y_test, pred_sk)
print(metrics_sk)
print("Train time:", train_time_sk)


# OOB curve
print("\n=== OOB Curve ===")
plot_oob_curve(X_train, y_train)