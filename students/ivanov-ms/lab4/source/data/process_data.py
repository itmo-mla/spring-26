import numpy as np
import pandas as pd
from typing import Optional, Tuple


class StandardScaler:
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray):
        self._mean = X.mean(axis=0, keepdims=True)
        self._std = X.std(axis=0, keepdims=True)

    def transform(self, X: np.ndarray):
        if self._mean is None or self._std is None:
            raise ValueError("StandardScaler wasn't fitted")
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray):
        if self._mean is None or self._std is None:
            raise ValueError("StandardScaler wasn't fitted")
        return X * self._std + self._mean


def prepare_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Prepare features for the dataset.
    - One-hot encode categorical features
    - Drop rows with missing values (if drop_na=True)
    """
    df = df.copy()

    # Identify target column
    target_col = 'class' if 'class' in df.columns else 'target'
    if target_col not in df.columns:
        raise ValueError(f"Target column not found. Columns: {list(df.columns)}")

    # Drop rows with any NaN if requested
    if drop_na:
        df = df.dropna()

    # Rename to 'target' for consistency
    df = df.rename(columns={target_col: 'target'})

    # Separate target from features
    y = df['target']
    X = df.drop(columns=['target'])

    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []

    for col in X.columns:
        dtype = X[col].dtype
        if dtype == 'object' or dtype == 'str' or dtype.name == 'category':
            categorical_cols.append(col)
        elif dtype in ['int64', 'float64']:
            # Low cardinality integers might be categorical
            unique_count = X[col].nunique()
            if unique_count <= 10 and dtype in ['int64']:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            numerical_cols.append(col)

    print(f"Detected {len(categorical_cols)} categorical columns: {categorical_cols}")
    print(f"Detected {len(numerical_cols)} numerical columns: {numerical_cols}")

    # One-hot encode categorical variables (keep all columns, no drop_first)
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False, dtype=float)

    if numerical_cols:
        scaler = StandardScaler()
        # Ensure columns exist after one-hot
        num_cols_present = [col for col in numerical_cols if col in X.columns]
        if num_cols_present:
            X[num_cols_present] = scaler.fit_transform(X[num_cols_present].to_numpy())

    # Convert boolean to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    # Combine with target
    X['target'] = y

    return X


def train_test_split(
        df: pd.DataFrame, train_size: float = 0.3,
        random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    targets_probs = df['target'].value_counts(normalize=True)
    probs = df['target'].map(targets_probs)
    probs /= probs.sum()

    rnd_indexes = rng.choice(df.shape[0], df.shape[0], replace=False, p=probs.to_numpy())

    split_lim = round(df.shape[0] * train_size)

    features_arr = df.drop('target', axis=1).to_numpy()
    target_arr = df['target'].to_numpy()

    columns = [col for col in df.columns if col != 'target']

    X_train, X_test = features_arr[rnd_indexes[:split_lim]], features_arr[rnd_indexes[split_lim:]]
    y_train, y_test = target_arr[rnd_indexes[:split_lim]], target_arr[rnd_indexes[split_lim:]]

    X_train = pd.DataFrame(data=X_train, columns=columns)
    X_test = pd.DataFrame(data=X_test, columns=columns)

    return X_train, X_test, y_train, y_test
