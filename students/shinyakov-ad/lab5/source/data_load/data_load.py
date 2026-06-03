import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_PATH_STRING = "mstz/speeddating"
DATASET_CONFIG = "dating"


def load_raw_dataframe() -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_PATH_STRING, DATASET_CONFIG, split="train")
    return dataset.to_pandas()


def make_ids(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = df[cols].astype(str).fillna("NA").to_numpy()
    keys = [tuple(row) for row in values]
    codes, _ = pd.factorize(pd.Index(keys))
    return codes


def prepare_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if {"iid", "pid", "match"}.issubset(df.columns):
        interactions = df[["iid", "pid", "match"]].dropna().copy()
        interactions["iid"] = interactions["iid"].astype(int)
        interactions["pid"] = interactions["pid"].astype(int)
        interactions["rating"] = interactions["match"].astype(float)
        return interactions[["iid", "pid", "rating"]]

    user_cols = [
        col
        for col in df.columns
        if col.startswith("dater_")
        or col.startswith("self_reported_")
        or col.endswith("_for_dater")
        or col in {"is_dater_male", "dater_age", "dater_race"}
    ]
    item_cols = [
        col
        for col in df.columns
        if col.startswith("dated_")
        or col.endswith("_for_dated")
        or col in {"dated_age", "dated_race"}
    ]

    prepared = df.copy()
    prepared["iid"] = make_ids(prepared, user_cols)
    prepared["pid"] = make_ids(prepared, item_cols)
    prepared["match"] = prepared["is_match"]

    interactions = prepared[["iid", "pid", "match"]].dropna().copy()
    interactions["iid"] = interactions["iid"].astype(int)
    interactions["pid"] = interactions["pid"].astype(int)
    interactions["rating"] = interactions["match"].astype(float)
    return interactions[["iid", "pid", "rating"]]


def build_matrix(interactions: pd.DataFrame, user_to_idx=None, item_to_idx=None):
    if user_to_idx is None:
        user_ids = sorted(interactions["iid"].unique())
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    if item_to_idx is None:
        item_ids = sorted(interactions["pid"].unique())
        item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    known = interactions[
        interactions["iid"].isin(user_to_idx)
        & interactions["pid"].isin(item_to_idx)
    ].copy()

    rows = known["iid"].map(user_to_idx).to_numpy()
    cols = known["pid"].map(item_to_idx).to_numpy()
    ratings = known["rating"].to_numpy(dtype=float)

    matrix = np.zeros((len(user_to_idx), len(item_to_idx)), dtype=float)
    matrix[rows, cols] = ratings

    return matrix, rows, cols, ratings, user_to_idx, item_to_idx


def load_dataset(test_size=0.2, random_state=42):
    df = load_raw_dataframe()
    interactions = prepare_interactions(df)

    train_interactions, test_interactions = train_test_split(
        interactions,
        test_size=test_size,
        random_state=random_state,
        stratify=interactions["rating"],
    )

    train_matrix, train_rows, train_cols, train_ratings, user_to_idx, item_to_idx = build_matrix(
        train_interactions
    )
    _, test_rows, test_cols, test_ratings, _, _ = build_matrix(
        test_interactions,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
    )

    return {
        "train_matrix": train_matrix,
        "train_rows": train_rows,
        "train_cols": train_cols,
        "train_ratings": train_ratings,
        "test_rows": test_rows,
        "test_cols": test_cols,
        "test_ratings": test_ratings,
        "interactions": interactions,
        "raw_dataframe": df,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
    }
