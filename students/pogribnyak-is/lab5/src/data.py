import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "ratings.csv"
    if cache.exists():
        return pd.read_csv(cache)

    print("Downloading MovieLens 100K...")
    r = requests.get(ML100K_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("ml-100k/u.data") as f:
            df = pd.read_csv(f, sep="\t", header=None,
                             names=["user_id", "item_id", "rating", "timestamp"])
    df.drop(columns="timestamp", inplace=True)
    df.to_csv(cache, index=False)
    print(f"Saved {len(df)} ratings to {cache}")
    return df


def build_matrix(df: pd.DataFrame):
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    v2i = {v: i for i, v in enumerate(items)}

    R = np.zeros((len(users), len(items)), dtype=np.float32)
    for row in df.itertuples(index=False):
        R[u2i[row.user_id], v2i[row.item_id]] = row.rating
    return R, u2i, v2i


def train_test_split(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_ratio)
    test_df = df.iloc[idx[:n_test]].reset_index(drop=True)
    train_df = df.iloc[idx[n_test:]].reset_index(drop=True)
    return train_df, test_df
