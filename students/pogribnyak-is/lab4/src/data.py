import numpy as np
import pandas as pd
from pathlib import Path

_DATASET = "vjchoudhary7/customer-segmentation-tutorial-in-python"
_CSV = "Mall_Customers.csv"
_FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]


def load_data() -> tuple[np.ndarray, str]:
    try:
        import kagglehub
        dataset_path = kagglehub.dataset_download(_DATASET)
        csv_path = Path(dataset_path) / _CSV
        df = pd.read_csv(csv_path)
        print(f"Mall Customers: {len(df)} samples  |  features: {_FEATURES}")
        return df[_FEATURES].values.astype(float), "Mall Customers"
    except Exception as e:
        print(f"kagglehub error: {type(e).__name__}: {e}")
        print("Kaggle unavailable — using synthetic data")
        return _synthetic(), "Synthetic (5 Gaussians)"


def _synthetic() -> np.ndarray:
    rng = np.random.default_rng(42)
    centers = [(20, 20), (20, 80), (55, 50), (80, 15), (80, 80)]
    cov = np.diag([50.0, 150.0])
    return np.clip(
        np.vstack([rng.multivariate_normal(c, cov, 40) for c in centers]),
        [0, 0], [137, 100],
    )
