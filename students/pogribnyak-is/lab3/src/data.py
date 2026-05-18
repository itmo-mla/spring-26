import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
XLSX = DATA_DIR / "Dry_Bean_Dataset.xlsx"


def _download():
    DATA_DIR.mkdir(exist_ok=True)
    if XLSX.exists():
        return
    print("Downloading Dry Bean Dataset from UCI...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            if name.lower().endswith(".xlsx"):
                data = z.read(name)
                XLSX.write_bytes(data)
                break
    print(f"  Saved to {XLSX}")


_le = LabelEncoder()


def load() -> tuple[np.ndarray, np.ndarray]:
    _download()
    df = pd.read_excel(XLSX)
    X = df.drop(columns=["Class"]).values.astype(np.float64)
    y = _le.fit_transform(df["Class"].values)
    return X, y


def get_class_names() -> list[str]:
    return list(_le.classes_)


