import os
import urllib.request
import tarfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


DATASET_URL = "http://konect.cc/files/download.tsv.libimseti.tar.bz2"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RATINGS_FILE = os.path.join(DATA_DIR, "libimseti", "out.libimseti")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RATINGS_FILE):
        return

    tar_path = os.path.join(DATA_DIR, "libimseti.tar.bz2")
    urllib.request.urlretrieve(DATASET_URL, tar_path)
    with tarfile.open(tar_path, "r:bz2") as t:
        t.extractall(DATA_DIR)


def load_ratings(min_ratings_per_user=20, min_ratings_per_item=20, sample_users=5000):
    """
    min_ratings_per_user: минимальное число оценок у пользователя
    min_ratings_per_item: минимальное число оценок у профиля
    sample_users: ограничение числа пользователей
    """
    download_dataset()
    rows = []
    with open(RATINGS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
    df["rating"] = df["rating"].astype(np.float32)

    print(f"Исходный размер датасета: {len(df):,} оценок, "
          f"{df['user_id'].nunique():,} пользователей, "
          f"{df['item_id'].nunique():,} профилей")

    # Фильтрация редких пользователей и айтемов
    for _ in range(3):
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_ratings_per_user].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_ratings_per_item].index)]

    # Сэмплирование пользователей для управляемого размера
    if sample_users is not None and df["user_id"].nunique() > sample_users:
        top_users = df["user_id"].value_counts().head(sample_users).index
        df = df[df["user_id"].isin(top_users)]

    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx).astype(np.int32)
    df["item_idx"] = df["item_id"].map(item2idx).astype(np.int32)

    n_users = len(user_ids)
    n_items = len(item_ids)
    density = len(df) / (n_users * n_items) * 100

    print(f"После фильтрации: {len(df):,} оценок, "
          f"{n_users:,} пользователей, {n_items:,} профилей, "
          f"плотность матрицы: {density:.3f}%")

    return df.reset_index(drop=True), n_users, n_items


def train_test_split(df, test_fraction=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    test_mask = np.zeros(len(df), dtype=bool)

    for _, group in df.groupby("user_idx"):
        n_test = max(1, int(len(group) * test_fraction))
        test_indices = rng.choice(group.index, size=n_test, replace=False)
        test_mask[test_indices] = True

    train_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    print(f"Train: {len(train_df):,} оценок | Test: {len(test_df):,} оценок")
    return train_df, test_df


def build_sparse_matrix(df, n_users, n_items):
    R = csr_matrix(
        (df["rating"].values, (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    return R