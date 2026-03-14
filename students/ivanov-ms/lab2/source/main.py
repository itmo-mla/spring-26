import argparse
import time
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest

from data import run_data_pipeline
from models import RandomForest
from utils import train_eval_model, compare_with_sklearn
from utils.plotting import plot_feature_importances, plot_roc_curve, plot_confusion_matrix, plot_learning_curve


def main():
    parser = argparse.ArgumentParser(description='Random Forest Classification Pipeline')
    parser.add_argument(
        '--train-size', type=float, default=0.7,
        help='Proportion of data for training (default: 0.7)'
    )
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--grid-search', action='store_true',
                        help='Perform grid search for hyperparameter tuning using OOB score')
    parser.add_argument('--max-estimators', type=int, default=500,
                        help='Max number of estimators for in model (default: 500)')
    parser.add_argument('--learning-curve', action='store_true',
                        help='Generate learning curve for varying n_estimators (up to final-n-estimators)')
    parser.add_argument('--with-plotting', action='store_true',
                        help='Save plots to images/ directory')

    args = parser.parse_args()

    # Data pipeline
    print("Running data pipeline...")
    X_train, X_test, y_train, y_test = run_data_pipeline(train_size=args.train_size, random_seed=args.random_seed)
    print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")

    # Baseline hyperparameters (will be tuned if grid-search)
    params = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 5,
        'max_features': 0.33
    }

    if args.grid_search:
        print("\nStarting grid search over hyperparameters using OOB score...")
        param_grid = {
            'n_estimators': [25, 50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.33, None]
        }
        best_score = -1
        best_params = None

        # Simple loop over some combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        for combo in product(*values):
            combo_params = dict(zip(keys, combo))
            combo_params['random_state'] = args.random_seed
            try:
                model = RandomForest(**combo_params)
                model.fit(X_train, y_train)
                score = model.oob_score_
                if score > best_score:
                    print(f"New best OOB Score: {score:.4f}, params: {combo_params}")
                    best_score = score
                    best_params = combo_params
            except Exception as e:
                print(f"Error with combo: {e}")
                continue

        print(f"\nBest OOB Score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        params.update(best_params)

    # Train custom Random Forest with final n_estimators
    print("\nTraining Custom Random Forest...")
    custom_rf = RandomForest(**params)
    cm_custom = train_eval_model(custom_rf, X_train, y_train, X_test, y_test, log_prefix="  ")
    print(f"  OOB Score: {custom_rf.oob_score_:.4f}" if custom_rf.oob_score_ else "")

    # Train sklearn Random Forest with same parameters
    print("\nTraining Sklearn Random Forest...")
    sklearn_rf = SklearnRandomForest(**params, bootstrap=True, n_jobs=1)

    cm_sklearn = train_eval_model(sklearn_rf, X_train, y_train, X_test, y_test, log_prefix="  ")
    if hasattr(sklearn_rf, 'oob_score_') and sklearn_rf.oob_score_:
        print(f"  OOB Score: {sklearn_rf.oob_score_:.4f}")

    # Compare models
    models_scores = compare_with_sklearn({
        "Custom RF": custom_rf,
        "Sklearn RF": sklearn_rf
    }, X_test, y_test)

    # Feature importances
    if custom_rf.feature_importances_ is not None:
        if args.with_plotting:
            plot_feature_importances(
                custom_rf.feature_importances_,
                feat_names=custom_rf.feature_names_,
                img_name="feature_importances_custom.png"
            )

        indices = np.argsort(custom_rf.feature_importances_)[::-1][:10]
        print("\nTop 10 Feature Importances (Custom RF):")
        for idx in indices:
            if custom_rf.feature_names_ is not None:
                print(f"  Feature {custom_rf.feature_names_[idx]}: {custom_rf.feature_importances_[idx]:.4f}")
            else:
                print(f"  Feature {idx}: {custom_rf.feature_importances_[idx]:.4f}")

    # Permutation importance (OOB^j)
    print("\nComputing OOB permutation feature importance...")
    perm_importance = custom_rf.compute_oob_permutation_importance(X_train, y_train)
    if args.with_plotting:
        plot_feature_importances(
            perm_importance, feat_names=custom_rf.feature_names_,
            img_name="permutation_importance_oob.png"
        )

    perm_idx = np.argsort(perm_importance)[::-1][:10]
    print("\nTop 10 OOB^j Permutation Importances:")
    for idx in perm_idx:
        if custom_rf.feature_names_ is not None:
            print(f"  Feature {custom_rf.feature_names_[idx]}: {perm_importance[idx]:.4f}")
        else:
            print(f"  Feature {idx}: {perm_importance[idx]:.4f}")

    # Learning curves
    if args.learning_curve:
        print("\nGenerating learning curve...")
        n_estimators_range = [1, 5, 10, 25] + list(range(50, args.max_estimators + 1, 50))
        oob_scores = []
        test_scores = []
        train_scores = []
        for n in n_estimators_range:
            print(f"  n_estimators = {n}")
            cur_params = params.copy()
            cur_params["n_estimators"] = n
            model = RandomForest(**cur_params)
            model.fit(X_train, y_train)

            oob_scores.append(model.oob_score_ if model.oob_score_ else np.nan)
            test_scores.append(model.score(X_test, y_test))
            train_scores.append(model.score(X_train, y_train))

        if args.with_plotting:
            plot_learning_curve(
                n_estimators_range, train_scores, test_scores,
                oob_scores=oob_scores, img_name="learning_curve_n_estimators.png"
            )

    # Confusion matrices and ROC curves
    if args.with_plotting:
        plot_confusion_matrix(cm_custom, title="Confusion Matrix (Custom RF)", img_name="cm_custom_rf.png")
        plot_confusion_matrix(cm_sklearn, title="Confusion Matrix (Sklearn RF)", img_name="cm_sklearn_rf.png")
        plot_roc_curve(y_test, models_scores, img_name="roc_curve_comparison_rf.png")

    print("\nAll done!")


if __name__ == "__main__":
    main()
