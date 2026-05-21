import time
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.metrics import accuracy_score

from utils.config_loader import load_config
from data.datasets import DATASETS
from models.random_forest import RandomForest
from utils.grid_search import GridSearchOOB
from utils.feature_importance import compute_feature_importance
from utils.plots import plot_conf_matrix, plot_feature_importance, plot_roc_curve


def main():
    config = load_config()
    project_root = Path(__file__).resolve().parent
    plots_dir = project_root / config.get("plots", {}).get("dir", "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = config['dataset']['name']
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASETS.keys())}")
    
    dataset_class = DATASETS[dataset_name]
    dataset = dataset_class(seed=config['dataset']['seed'])
    dataset.split_and_scale(
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size']
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"Train shape: {dataset.X_train.shape}")
    print(f"Val shape: {dataset.X_val.shape}")
    print(f"Test shape: {dataset.X_test.shape}")
    print()

    param_grid = config['model']['grid_search']

    base_estimator = RandomForest(
        random_state=config['model']['random_state'],
        n_jobs=config['model']['n_jobs'],
        oob_score=True
    )

    print("Performing Grid Search with OOB scoring...")
    grid_search = GridSearchOOB(base_estimator, param_grid)

    start_time = time.time()
    grid_search.fit(dataset.X_train, dataset.y_train)
    grid_search_time = time.time() - start_time

    best_params = grid_search.best_params_

    best_model = RandomForest(
        **best_params,
        random_state=config['model']['random_state'],
        n_jobs=config['model']['n_jobs'],
        oob_score=True
    )

    start_time = time.time()
    best_model.fit(dataset.X_train, dataset.y_train)
    training_time = time.time() - start_time
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best OOB score: {grid_search.best_score_:.4f}")
    print(f"GridSearch time: {grid_search_time:.2f}s")
    print(f"Best model fit time: {training_time:.2f}s")
    print()

    print("Computing feature importance via OOB^j...")
    feature_importance = compute_feature_importance(best_model, dataset.X_train, dataset.y_train)
    print(f"Top 5 features: {np.argsort(feature_importance)[-5:][::-1]}")
    print()

    y_pred = best_model.predict(dataset.X_test)
    test_accuracy = accuracy_score(dataset.y_test, y_pred)
    print(f"Test accuracy (custom): {test_accuracy:.4f}")
    print()

    feature_names = list(dataset.df.drop(columns=[dataset.target_col]).columns) if hasattr(dataset, "df") else None
    plot_feature_importance(
        feature_importance,
        top_k=min(20, len(feature_importance)),
        title="Custom RF Feature Importance (OOB^j)",
        out_dir=plots_dir,
        filename="feature_importance_custom.png",
        feature_names=feature_names,
    )
    plot_conf_matrix(
        dataset.y_test,
        y_pred,
        "Custom RF Confusion Matrix",
        out_dir=plots_dir,
        filename="confusion_matrix_custom.png",
    )
    plot_roc_curve(
        best_model,
        dataset.X_test,
        dataset.y_test,
        title="Custom RF ROC Curve",
        out_dir=plots_dir,
        filename="roc_curve_custom.png",
    )

    print("Comparing with sklearn RandomForest...")
    sklearn_params = config['comparison']['sklearn_model']
    sklearn_model = SklearnRandomForest(**sklearn_params, oob_score=True)

    y_train_sklearn = (dataset.y_train + 1) // 2
    y_test_sklearn = (dataset.y_test + 1) // 2
    
    start_time = time.time()
    sklearn_model.fit(dataset.X_train, y_train_sklearn)
    sklearn_training_time = time.time() - start_time
    
    sklearn_test_pred = sklearn_model.predict(dataset.X_test)
    sklearn_test_pred = sklearn_test_pred * 2 - 1
    sklearn_test_accuracy = accuracy_score(dataset.y_test, sklearn_test_pred)
    
    print(f"Sklearn OOB score: {sklearn_model.oob_score_:.4f}")
    print(f"Sklearn test accuracy: {sklearn_test_accuracy:.4f}")
    print(f"Sklearn training time: {sklearn_training_time:.2f}s")
    print()

    plot_conf_matrix(
        dataset.y_test,
        sklearn_test_pred,
        "Sklearn RF Confusion Matrix",
        out_dir=plots_dir,
        filename="confusion_matrix_sklearn.png",
    )
    plot_roc_curve(
        sklearn_model,
        dataset.X_test,
        y_test_sklearn,
        title="Sklearn RF ROC Curve",
        out_dir=plots_dir,
        filename="roc_curve_sklearn.png",
    )

    print("=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<30} {'Custom':<15} {'Sklearn':<15}")
    print("-" * 50)
    print(f"{'OOB Score':<30} {best_model._oob_score_:<15.4f} {sklearn_model.oob_score_:<15.4f}")
    print(f"{'Test Accuracy':<30} {test_accuracy:<15.4f} {sklearn_test_accuracy:<15.4f}")
    print(f"{'Best Fit Time (s)':<30} {training_time:<15.2f} {sklearn_training_time:<15.2f}")
    print(f"{'GridSearch Time (s)':<30} {grid_search_time:<15.2f} {'-':<15}")
    print("=" * 50)


if __name__ == "__main__":
    main()
