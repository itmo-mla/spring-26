import argparse
import time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoosting

from data import run_data_pipeline
from models import GradientBoostingClassifier
from utils import train_eval_model, compare_with_sklearn
from utils.plotting import plot_roc_curve, plot_confusion_matrix, plot_feature_importances, plot_learning_curve_iterations


def main():
    parser = argparse.ArgumentParser(description='Gradient Boosting Classification Pipeline')
    parser.add_argument(
        '--train-size', type=float, default=0.7,
        help='Proportion of data for training (default: 0.7)'
    )
    parser.add_argument(
        '--random-seed', type=int, default=42, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--n-estimators', type=int, default=100, help='Number of boosting stages (default: 100)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.1, help='Learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--max-depth', type=int, default=3, help='Maximum depth of base learners (default: 3)'
    )
    parser.add_argument(
        '--min-samples-split', type=int, default=2, help='Minimum samples required to split (default: 2)'
    )
    parser.add_argument(
        '--subsample', type=float, default=1.0, help='Subsample ratio (default: 1.0)')
    parser.add_argument(
        '--with-plotting', action='store_true', help='Save plots to images/ directory'
    )

    args = parser.parse_args()

    # Data pipeline
    X_train, X_test, y_train, y_test = run_data_pipeline(
        train_size=args.train_size,
        random_seed=args.random_seed
    )
    print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")

    # Train custom Gradient Boosting
    print("\n" + "=" * 60)
    print("TRAINING CUSTOM GRADIENT BOOSTING")
    print("=" * 60)

    custom_gb = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        subsample=args.subsample,
        random_state=args.random_seed
    )

    print(f"Custom GB: {custom_gb}")

    cm_custom = train_eval_model(custom_gb, X_train, y_train, X_test, y_test, log_prefix=" ")

    # Train sklearn Gradient Boosting for comparison
    print("\n" + "=" * 60)
    print("TRAINING SKLEARN GRADIENT BOOSTING")
    print("=" * 60)

    sklearn_gb = SklearnGradientBoosting(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        subsample=args.subsample,
        random_state=args.random_seed
    )

    print(f"Sklearn GB: {sklearn_gb}")

    cm_sklearn = train_eval_model(sklearn_gb, X_train, y_train, X_test, y_test, log_prefix=" ")

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    models_scores = compare_with_sklearn({
        "Custom GB": custom_gb,
        "Sklearn GB": sklearn_gb
    }, X_test, y_test)

    # Feature importances
    if custom_gb.feature_importances_ is not None:
        if args.with_plotting:
            plot_feature_importances(
                custom_gb.feature_importances_,
                feat_names=custom_gb.feature_names_,
                img_name="feature_importances_custom.png"
            )

        indices = np.argsort(custom_gb.feature_importances_)[::-1][:10]
        print("\nTop 10 Feature Importances (Custom GB):")
        for idx in indices:
            if custom_gb.feature_names_ is not None:
                print(f"  Feature {custom_gb.feature_names_[idx]}: {custom_gb.feature_importances_[idx]:.4f}")
            else:
                print(f"  Feature {idx}: {custom_gb.feature_importances_[idx]:.4f}")

    if sklearn_gb.feature_importances_ is not None:
        feature_names_ = list(X_train.columns)
        if args.with_plotting:
            plot_feature_importances(
                sklearn_gb.feature_importances_,
                feat_names=feature_names_,
                img_name="feature_importances_sklearn.png"
            )

        indices = np.argsort(sklearn_gb.feature_importances_)[::-1][:10]
        print("\nTop 10 Feature Importances (Sklearn GB):")
        for idx in indices:
            print(f"  Feature {feature_names_[idx]}: {sklearn_gb.feature_importances_[idx]:.4f}")

    # Plot learning curve (training accuracy vs iterations)
    if args.with_plotting:
        if hasattr(custom_gb, 'train_loss_') and custom_gb.train_loss_:
            plot_learning_curve_iterations(
                custom_gb.train_loss_, name="Loss", img_name="learning_curve_loss.png"
            )
        if hasattr(custom_gb, 'train_auc_scores_'):
            plot_learning_curve_iterations(
                custom_gb.train_auc_scores_, name="AUC", img_name="learning_curve_auc.png"
            )
        plot_confusion_matrix(cm_custom, title="Confusion Matrix (Custom GB)", img_name="cm_custom_gb.png")
        plot_confusion_matrix(cm_sklearn, title="Confusion Matrix (Sklearn GB)", img_name="cm_sklearn_gb.png")
        plot_roc_curve(y_test, models_scores, img_name="roc_curve_comparison_gb.png")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
