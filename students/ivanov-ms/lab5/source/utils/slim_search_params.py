import warnings
import numpy as np

from data import run_data_pipeline
from models import ReferenceSLIM
from utils.compare import train_eval_model
from utils.utils import Columns

warnings.filterwarnings('ignore')


def search_slim_params():
    # Data pipeline
    X_train, X_test = run_data_pipeline(train_size=0.7)
    print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    print("Create matrices")
    X_train_matrix = X_train.pivot(index=Columns.User, columns=Columns.Item, values=Columns.Rating)
    X_train_matrix = X_train_matrix[sorted(X_train_matrix.columns)]
    X_train_matrix = X_train_matrix.fillna(0)
    print(f"Train matrix created: {X_train_matrix.shape}")

    # Train reference ALS
    print("\n" + "=" * 60)
    print("TRAINING REFERENCE ALS")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("TRAINING REFERENCE SLIM")
    print("=" * 60)
    l1_params = np.linspace(0.1, 1, 10)
    l2_params = np.linspace(0.1, 1, 10)
    l1, l2 = np.meshgrid(l1_params, l2_params)
    l1, l2 = l1.flatten(), l2.flatten()

    best_score, best_params = None, None

    for i, (l1_coef, l2_coef) in enumerate(zip(l1, l2)):
        reference_slim = ReferenceSLIM(
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            n_processes=8,
            max_iter=5000,
            tol=1e-5,
        )
        reference_slim_topk, reference_slim_metrics = train_eval_model(
            reference_slim, X_train, X_train_matrix, X_test, k=10
        )
        score = reference_slim_metrics["NDCG@10"]
        if best_score is None or score > best_score:
            print(f"Step {i}. New best score: {score:.4f} with params l1={l1_coef:.2}, l2={l2_coef:.2}")
            best_score = score
            best_params = (l1_coef, l2_coef)

    print(f"\nDone search, best score: {best_score:.4f} with params {best_params}")


if __name__ == "__main__":
    search_slim_params()
