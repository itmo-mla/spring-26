from __future__ import annotations

from dataclasses import dataclass
import ssl
from urllib.error import URLError

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


CATEGORIES = [
    "comp.graphics",
    "rec.autos",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.misc",
]


@dataclass(frozen=True)
class TextMatrixBundle:
    documents: list[str]
    topics: list[str]
    topic_ids: np.ndarray
    matrix: np.ndarray
    feature_names: list[str]
    description: dict[str, int | float | str]


@dataclass(frozen=True)
class MatrixSplit:
    train_matrix: np.ndarray
    test_entries: pd.DataFrame
    density: float


def fetch_real_text_corpus(
    docs_per_category: int = 60,
    max_features: int = 220,
    random_state: int = 42,
) -> TextMatrixBundle:
    """Load a real 20 Newsgroups subset and convert it to a TF-IDF matrix."""
    dataset = _fetch_20newsgroups_subset()
    rng = np.random.RandomState(random_state)
    selected_indexes: list[int] = []

    targets = np.asarray(dataset.target, dtype=int)
    for target_id in range(len(dataset.target_names)):
        indexes = np.flatnonzero(targets == target_id)
        take = min(docs_per_category, len(indexes))
        selected_indexes.extend(rng.choice(indexes, size=take, replace=False).tolist())
    rng.shuffle(selected_indexes)

    documents = [dataset.data[index] for index in selected_indexes]
    topic_ids = targets[selected_indexes]
    topics = [dataset.target_names[target_id] for target_id in topic_ids]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.65,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        norm="l2",
    )
    matrix = vectorizer.fit_transform(documents).astype(float).toarray()
    feature_names = vectorizer.get_feature_names_out().tolist()
    density = float(np.count_nonzero(matrix) / matrix.size)

    return TextMatrixBundle(
        documents=documents,
        topics=topics,
        topic_ids=np.asarray(topic_ids, dtype=int),
        matrix=matrix,
        feature_names=feature_names,
        description={
            "name": "20 Newsgroups subset",
            "source": "sklearn.datasets.fetch_20newsgroups",
            "documents": len(documents),
            "topics": len(set(topics)),
            "terms": len(feature_names),
            "density": density,
            "rating": "TF-IDF(document, term)",
        },
    )


def _fetch_20newsgroups_subset():
    kwargs = {
        "subset": "train",
        "categories": CATEGORIES,
        "remove": ("headers", "footers", "quotes"),
        "shuffle": True,
        "random_state": 42,
    }
    try:
        return fetch_20newsgroups(**kwargs)
    except URLError as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error):
            raise
        ssl._create_default_https_context = ssl._create_unverified_context
        return fetch_20newsgroups(**kwargs)


def train_test_holdout(
    matrix: np.ndarray,
    test_fraction: float = 0.2,
    min_train_terms: int = 4,
    random_state: int = 42,
) -> MatrixSplit:
    rng = np.random.RandomState(random_state)
    train = matrix.copy()
    rows: list[dict[str, float | int]] = []

    for user_id in range(matrix.shape[0]):
        nonzero_items = np.flatnonzero(matrix[user_id] > 0)
        if len(nonzero_items) <= min_train_terms:
            continue
        n_test = max(1, int(round(len(nonzero_items) * test_fraction)))
        n_test = min(n_test, len(nonzero_items) - min_train_terms)
        if n_test <= 0:
            continue
        test_items = rng.choice(nonzero_items, size=n_test, replace=False)
        for item_id in test_items:
            rows.append(
                {
                    "user": user_id,
                    "item": int(item_id),
                    "rating": float(matrix[user_id, item_id]),
                }
            )
        train[user_id, test_items] = 0.0

    density = float(np.count_nonzero(train) / train.size)
    return MatrixSplit(
        train_matrix=train,
        test_entries=pd.DataFrame(rows),
        density=density,
    )
