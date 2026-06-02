from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TextDataset:
    matrix: sparse.csr_matrix
    document_ids: np.ndarray
    feature_names: np.ndarray
    category_names: tuple[str, ...]
    raw_categories: np.ndarray


def load_text_interaction_matrix(
    *,
    categories: tuple[str, ...] = ("sci.med", "sci.space"),
    max_features: int = 400,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[TextDataset, sparse.csr_matrix, sparse.csr_matrix]:
    corpus = fetch_20newsgroups(
        subset="all",
        categories=list(categories),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=random_state,
    )
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=2)
    matrix = vectorizer.fit_transform(corpus.data).tocsr().astype(np.float32)
    document_ids = np.arange(matrix.shape[0])
    dataset = TextDataset(
        matrix=matrix,
        document_ids=document_ids,
        feature_names=vectorizer.get_feature_names_out(),
        category_names=categories,
        raw_categories=np.asarray(corpus.target),
    )
    train, test = train_test_split_matrix(
        matrix, test_size=test_size, random_state=random_state
    )
    return dataset, train, test


def train_test_split_matrix(
    matrix: sparse.csr_matrix,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    rng = np.random.default_rng(random_state)
    train = matrix.copy().tocsr()
    test = sparse.csr_matrix(train.shape, dtype=np.float32)

    rows, cols = train.nonzero()
    n_holdout = int(len(rows) * test_size)
    if n_holdout == 0 and len(rows) > 0:
        n_holdout = 1
    holdout_idx = rng.choice(len(rows), size=n_holdout, replace=False)

    test_rows = rows[holdout_idx]
    test_cols = cols[holdout_idx]
    test_values = train.data[holdout_idx].copy()

    train.data[holdout_idx] = 0.0
    train.eliminate_zeros()
    test = sparse.csr_matrix(
        (test_values, (test_rows, test_cols)),
        shape=train.shape,
        dtype=np.float32,
    )
    return train, test
