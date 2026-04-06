import kagglehub
import pandas as pd
import os

DATASET_NAME = 'taweilo/loan-approval-classification-data'
DATA_FILENAME = "loan_approval.csv"  # Might be actual filename; adjust if needed


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def load_data() -> pd.DataFrame:
    """
    Download and load the Loan Approval Classification dataset from Kaggle.

    Returns:
        pandas DataFrame with the dataset
    """
    path = kagglehub.dataset_download(DATASET_NAME)
    # Find the csv file in the downloaded directory
    files = os.listdir(path)
    csv_file = None
    for f in files:
        if f.lower().endswith('.csv'):
            csv_file = f
            break
    if csv_file is None:
        raise FileNotFoundError(f"No CSV file found in {path}")
    file_path = os.path.join(path, csv_file)
    return load_data_from_csv(file_path)
