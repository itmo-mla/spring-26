from typing import Optional
import time

from .load_data import load_data
from .process_data import prepare_features, train_test_split, filter_train_test


def run_data_pipeline(
    return_split: bool = True,
    train_size: float = 0.7,
    save_path: Optional[str] = None
):
    """
    Run all data preprocessing pipeline:
        1. loading data
        2. Preprocessing
        3. Splitting into train/test (if return_split=True)

    :param return_split: If True, return split arrays; else return full DataFrame
    :param train_size: Proportion for training set
    :param save_path: Optional path to save processed CSV

    :return: If return_split: X_train, X_test
            Else: processed DataFrame
    """
    print("Running data pipeline...")
    start_time = time.time()

    # Load raw data
    df = load_data()

    # Prepare features (encoding, scaling)
    df = prepare_features(df)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    if return_split:
        result = train_test_split(df, train_size=train_size)
        result = filter_train_test(*result)
    else:
        result = df

    print(f"Data pipeline finished in {time.time() - start_time:.2f} sec")
    return result
