import argparse
from sklearn.mixture import GaussianMixture

from data import run_data_pipeline
from models import GaussianMixtureModel
from utils import train_eval_model


def main():
    parser = argparse.ArgumentParser(description='Gaussian Mixture Model Pipeline')
    parser.add_argument(
        '-n', type=int, default=2, help='Number of components'
    )
    parser.add_argument(
        '--train-size', type=float, default=0.7,
        help='Proportion of data for training (default: 0.7)'
    )
    parser.add_argument(
        '--random-seed', type=int, default=42, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-iter', type=int, default=100, help='Number of train iterations'
    )
    parser.add_argument(
        '--tol', type=float, default=1e-4, help='Train tolerance'
    )
    parser.add_argument(
        '--init-params', type=str, default="kmeans", help='How to init params (kmeans or random)'
    )
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

    # Train custom GMM
    print("\n" + "=" * 60)
    print("TRAINING CUSTOM GMM")
    print("=" * 60)

    custom_gmm = GaussianMixtureModel(
        n_components=args.n,
        max_iter=args.max_iter,
        tol=args.tol,
        init_params=args.init_params
    )

    print(f"Custom GB: {custom_gmm}")

    train_eval_model(custom_gmm, X_train, X_test)

    # Train sklearn Gradient Boosting for comparison
    print("\n" + "=" * 60)
    print("TRAINING SKLEARN GAUSSIAN MIXTURE")
    print("=" * 60)

    sklearn_gmm = GaussianMixture(
        n_components=args.n,
        max_iter=args.max_iter,
        tol=args.tol,
        init_params=args.init_params,
        random_state=args.random_seed
    )

    print(f"Sklearn GB: {sklearn_gmm}")

    train_eval_model(sklearn_gmm, X_train, X_test)

    # Plot learning curve (training accuracy vs iterations)
    if args.with_plotting:
        from utils.plotting import plot_learning_curve_iterations, plot_pca

        if hasattr(custom_gmm, 'train_ll_hist_') and custom_gmm.train_ll_hist_:
            plot_learning_curve_iterations(
                custom_gmm.train_ll_hist_, name="Log-Likelihood", img_name="learning_curve_ll.png"
            )

        plot_pca(custom_gmm, X_train, X_test, y_train, y_test, img_name="custom_gmm_pca.png")
        plot_pca(sklearn_gmm, X_train, X_test, y_train, y_test, img_name="sklearn_gmm_pca.png")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
