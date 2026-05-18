import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as SklearnGB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.boosting import GradientBoostingClassifier
from src.data import get_class_names, load
from src.evaluation import cross_validate
import src.plots as plots

N_ESTIMATORS = 100
LR = 0.1


def main():

    X, y = load()
    class_names = get_class_names()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes\n")

    custom_fn = lambda: GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=LR)
    sklearn_fn = lambda: SklearnGB(n_estimators=N_ESTIMATORS, learning_rate=LR, max_depth=1)

    print("Custom Gradient Boosting (5-fold CV)")
    custom = cross_validate(custom_fn, X, y)
    print(f"  accuracy {custom['accuracy'].mean():.4f}")
    print(f"  F1 macro {custom['f1'].mean():.4f}")
    print(f"  time     {custom['time'].mean():.2f}s / fold\n")

    print("Sklearn GradientBoostingClassifier (5-fold CV)")
    sk = cross_validate(sklearn_fn, X, y)
    print(f"  accuracy {sk['accuracy'].mean():.4f}")
    print(f"  F1 macro {sk['f1'].mean():.4f}")
    print(f"  time     {sk['time'].mean():.2f}s / fold\n")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    full_model = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=LR)
    full_model.fit(Xs, y)
    cm = confusion_matrix(y, full_model.predict(Xs))

    plots.learning_curve(full_model.train_scores_)
    plots.cv_comparison(custom["accuracy"], sk["accuracy"])
    plots.time_comparison(custom["time"].mean(), sk["time"].mean())
    plots.confusion_matrix(cm, class_names)

    print("\nSummary")
    print(f"Custom  accuracy: {custom['accuracy'].mean():.4f}  |  time: {custom['time'].mean():.2f}s")
    print(f"Sklearn accuracy: {sk['accuracy'].mean():.4f}  |  time: {sk['time'].mean():.2f}s")


if __name__ == "__main__":
    main()
