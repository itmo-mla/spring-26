import argparse
import warnings
import pandas as pd

from data import run_data_pipeline
from models import SLIM, ALS, ReferenceSLIM, ReferenceALS
from utils.compare import train_eval_model
from utils.utils import Columns

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Gaussian Mixture Model Pipeline')
    parser.add_argument(
        '--n-processes', type=int, default=1, help='Number of processes to parallel'
    )
    # SLIM params
    parser.add_argument(
        '--slim-l1', type=float, default=0.1, help='SLIM L1 Reg. term'
    )
    parser.add_argument(
        '--slim-l2', type=float, default=0.1, help='SLIM L2 Reg. term'
    )
    parser.add_argument(
        '--slim-max-iter', type=int, default=500, help='SLIM max iterations for ElasticNet'
    )
    parser.add_argument(
        '--slim-tol', type=float, default=1e-5, help='SLIM tolerance for ElasticNet'
    )
    parser.add_argument(
        '--slim-positive', action='store_true', help='SLIM use positive matrix for ElasticNet'
    )
    parser.add_argument(
        '--slim-selection', type=str, default='cyclic', choices=['cyclic', 'random'],
        help='SLIM selection for ElasticNet'
    )
    parser.add_argument(
        '--slim-use-mask', action='store_true',
        help='SLIM use masked training (only on positive samples + sampled `sample_perc` zeros)'
    )
    parser.add_argument(
        '--slim-sample-perc', type=float, default=1,
        help='SLIM zeros sample ratio (ratio to positive samples). Only works if --slim-use-mask provided'
    )
    # ALS params
    parser.add_argument(
        '--als-factors', type=int, default=20, help='ALS number of latent factors'
    )
    parser.add_argument(
        '--als-l2', type=float, default=0.1, help='ALS L2 Reg. Term'
    )
    parser.add_argument(
        '--als-epochs', type=int, default=50, help='ALS number of epochs to train'
    )
    parser.add_argument(
        '--als-batch-size', type=int, default=32, help='ALS batch size for parallel processing'
    )
    # General params
    parser.add_argument(
        '-k', type=int, default=10, help='`k` for with metrics will be calculated'
    )
    parser.add_argument(
        '--train-size', type=float, default=0.7,
        help='Proportion of data for training (default: 0.7)'
    )
    parser.add_argument(
        '--with-plotting', action='store_true', help='Save plots to images/ directory'
    )

    args = parser.parse_args()

    # Data pipeline
    X_train, X_test = run_data_pipeline(train_size=args.train_size)
    print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")

    print("Create matrices")
    X_train_matrix = X_train.pivot(index=Columns.User, columns=Columns.Item, values=Columns.Rating)
    X_train_matrix = X_train_matrix[sorted(X_train_matrix.columns)]
    X_train_matrix = X_train_matrix.fillna(0)
    print(f"Train matrix created: {X_train_matrix.shape}")

    # Train custom SLIM
    print("\n" + "=" * 60)
    print("TRAINING CUSTOM SLIM")
    print("=" * 60)

    custom_slim = SLIM(
        l1_coef=args.slim_l1,
        l2_coef=args.slim_l2,
        n_processes=args.n_processes,
        max_iter=args.slim_max_iter,
        tol=args.slim_tol,
        positive=args.slim_positive,
        selection=args.slim_selection,
        use_mask=args.slim_use_mask,
        sample_perc=args.slim_sample_perc
    )

    print(f"Custom SLIM: {custom_slim}")

    custom_slim_topk, custom_slim_metrics = train_eval_model(custom_slim, X_train, X_train_matrix, X_test, k=args.k)

    # Train custom ALS
    print("\n" + "=" * 60)
    print("TRAINING CUSTOM ALS")
    print("=" * 60)

    custom_als = ALS(
        n_processes=args.n_processes,
        n_factors=args.als_factors,
        lambda_reg=args.als_l2,
        n_epochs=args.als_epochs,
        batch_size=args.als_batch_size
    )

    print(f"Custom ALS: {custom_als}")

    custom_als_topk, custom_als_metrics = train_eval_model(custom_als, X_train, X_train_matrix, X_test, k=args.k)

    # Train reference SLIM
    print("\n" + "=" * 60)
    print("TRAINING REFERENCE SLIM")
    print("=" * 60)
    reference_slim = ReferenceSLIM(
        l1_coef=args.slim_l1,
        l2_coef=args.slim_l2,
        n_processes=args.n_processes,
        max_iter=args.slim_max_iter,
        tol=args.slim_tol,
    )

    reference_slim_topk, reference_slim_metrics = train_eval_model(
        reference_slim, X_train, X_train_matrix, X_test, k=args.k
    )

    # Train reference ALS
    print("\n" + "=" * 60)
    print("TRAINING REFERENCE ALS")
    print("=" * 60)

    reference_als = ReferenceALS(
        n_processes=args.n_processes,
        n_factors=args.als_factors,
        lambda_reg=args.als_l2,
        n_epochs=args.als_epochs,
    )

    reference_als_topk, reference_als_metrics = train_eval_model(
        reference_als, X_train, X_train_matrix, X_test, k=args.k
    )

    combined_metrics = {
        "Custom SLIM": custom_slim_metrics,
        "Custom ALS": custom_als_metrics,
        "Reference SLIM": reference_slim_metrics,
        "Reference ALS": reference_als_metrics
    }
    combined_metrics = pd.DataFrame(combined_metrics).T
    print("All metrics compare:")
    print(combined_metrics)

    # Plot learning curve (training accuracy vs iterations)
    if args.with_plotting:
        from utils.plotting import plot_learning_curve_iterations

        if hasattr(custom_als, 'train_history') and custom_als.train_history:
            plot_learning_curve_iterations(
                custom_als.train_history, name="ALS", img_name="learning_curve_als.png"
            )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
