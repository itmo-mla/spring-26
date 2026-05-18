from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

from gradient_boosting import GradientBoostingClassifier


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Custom gradient boosting")
    custom_model = GradientBoostingClassifier(**params)
    res_custom = cross_validate(
        custom_model, X, y,
        cv=cv,
        scoring='accuracy',
        return_train_score=False,
        n_jobs=1
    )
    acc_custom = res_custom['test_score']
    print(f"   Accuracy : {acc_custom.mean():.4f} +- {acc_custom.std():.4f}")
    print(f"   Fit Time : {res_custom['fit_time'].mean():.4f} s / fold")

    print("Sklearn gradient boosting")
    sklearn_model = SklearnGBClassifier(**params)
    res_sklearn = cross_validate(
        sklearn_model, X, y,
        cv=cv,
        scoring='accuracy',
        return_train_score=False,
        n_jobs=1
    )
    acc_sklearn = res_sklearn['test_score']
    print(f"   Accuracy : {acc_sklearn.mean():.4f} +- {acc_sklearn.std():.4f}")
    print(f"   Fit Time : {res_sklearn['fit_time'].mean():.4f} s / fold")


if __name__ == "__main__":
    main()