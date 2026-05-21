import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from gradient_boosting import GradientBoostingRegressorCustom
from utils import measure_time


def cross_val_custom(model_class, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        model = model_class(n_estimators=50)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        mse = mean_squared_error(y[test_idx], pred)
        scores.append(mse)

    return np.mean(scores)


def main():
    data = fetch_california_housing()
    X, y = data.data, data.target

    # --- train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Custom ---
    custom_model = GradientBoostingRegressorCustom(n_estimators=50)

    _, time_custom = measure_time(custom_model.fit, X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)
    mse_custom = mean_squared_error(y_test, y_pred_custom)

    # --- Sklearn ---
    sklearn_model = GradientBoostingRegressor(n_estimators=50)

    _, time_sklearn = measure_time(sklearn_model.fit, X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

    print("Custom GB:")
    print("MSE:", mse_custom)
    print("Time:", time_custom)

    print("\nSklearn GB:")
    print("MSE:", mse_sklearn)
    print("Time:", time_sklearn)

    # --- Cross-validation ---
    cv_custom = cross_val_custom(GradientBoostingRegressorCustom, X, y)
    print("\nCross-validation MSE (custom):", cv_custom)

    scores_sklearn = cross_val_score(
        GradientBoostingRegressor(n_estimators=50),
        X,
        y,
        scoring='neg_mean_squared_error',
        cv=5
    )

    print("Cross-validation MSE (sklearn):", -scores_sklearn.mean())

    # --- График ---
    plt.plot(custom_model.errors, label="Custom")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Training error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()