import time

from models.my_boosting import get_my_boosting
from models.sklearn_boosting import get_sklearn_boosting
from utils.dataset import load_titanic
from utils.metrics import evaluate_model
from utils.report import make_report, save_results



def train_model(train_function, X_train, y_train):
    start_time = time.perf_counter()
    grid_search = train_function(X_train, y_train)
    train_time = time.perf_counter() - start_time
    return grid_search, train_time



def build_result(model_name, grid_search, X_test, y_test):
    metrics = evaluate_model(model_name, grid_search.best_estimator_, X_test, y_test)
    return {
        "best_params_": grid_search.best_params_,
        "best_score_": grid_search.best_score_,
        "metrics": metrics,
    }



def main():
    X_train, X_test, y_train, y_test, _ = load_titanic()

    my_grid, my_time = train_model(get_my_boosting, X_train, y_train)
    sklearn_grid, sklearn_time = train_model(get_sklearn_boosting, X_train, y_train)

    my_result = build_result("My GradientBoosting", my_grid, X_test, y_test)
    sklearn_result = build_result(
        "Sklearn GradientBoostingClassifier",
        sklearn_grid,
        X_test,
        y_test,
    )

    make_report(my_result, sklearn_result, my_time, sklearn_time)
    save_results(my_result, sklearn_result, my_time, sklearn_time)


if __name__ == "__main__":
    main()
