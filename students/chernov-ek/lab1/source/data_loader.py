import requests
from pathlib import Path
import zipfile
import pandas as pd


ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"


def load_archive():
    output_file = DATA_PATH / "diabetes-prediction-dataset.zip"
    if output_file.exists():
        return

    url = "https://www.kaggle.com/api/v1/datasets/download/iammustafatz/diabetes-prediction-dataset"
    response = requests.get(url, stream=True, allow_redirects=True)

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def load_df():
    archive_path = DATA_PATH / "diabetes-prediction-dataset.zip"
    with zipfile.ZipFile(archive_path, "r") as z:
        with z.open("diabetes_prediction_dataset.csv") as f:
            df = pd.read_csv(f)

    return df
