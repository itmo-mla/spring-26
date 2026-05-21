import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from ensemble import CustomRandomForest


def oob_scorer(estimator, X, y):
    return getattr(estimator, 'oob_score_', 0.0)


def main():

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    param_grid = {
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'max_depth': [10, 50]
    }

    grid_search = GridSearchCV(
        estimator=CustomRandomForest(random_state=42),
        param_grid=param_grid,
        scoring=oob_scorer,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Лучшие параметры по OOB: {best_params}")
    print(f"Лучшая OOB-оценка (accuracy): {grid_search.best_score_:.4f}")

    t0 = time.time()
    custom_rf = CustomRandomForest(**best_params, random_state=42)
    custom_rf.fit(X_train, y_train)
    time_custom = time.time() - t0
    acc_custom = accuracy_score(y_test, custom_rf.predict(X_test))

    t0 = time.time()
    sklearn_rf = RandomForestClassifier(**best_params, oob_score=True, random_state=42, n_jobs=-1)
    sklearn_rf.fit(X_train, y_train)
    time_sklearn = time.time() - t0
    acc_sklearn = accuracy_score(y_test, sklearn_rf.predict(X_test))

    print(f"\n{'Метрика':<25} | {'Custom RSM/RF':<15} | {'Sklearn RF':<15}")
    print(f"{'Точность (Test)':<25} | {acc_custom:<15.4f} | {acc_sklearn:<15.4f}")
    print(f"{'OOB-оценка':<25} | {custom_rf.oob_score_:<15.4f} | {sklearn_rf.oob_score_:<15.4f}")
    print(f"{'Время обучения (сек)':<25} | {time_custom:<15.4f} | {time_sklearn:<15.4f}")

    print("\nТоп важных признаков:")
    importances = custom_rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. {data.feature_names[idx]:<20} | Важность: {importances[idx]:.2f}%")


if __name__ == "__main__":
    main()