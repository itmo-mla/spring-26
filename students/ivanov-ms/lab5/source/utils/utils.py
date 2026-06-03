import os
from typing import Dict, Any
from contextlib import contextmanager

import tempfile
import numpy as np
from dataclasses import dataclass


@dataclass
class Columns:
    User: str = "user_id"
    Item: str = "item_id"
    Rating: str = "rating"
    Score: str = "score"
    Rank: str = 'rank'
    Datetime: str = 'dt'


@contextmanager
def save_to_mmap(X: np.ndarray, name: str) -> Dict[str, Any]:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    X_path = tmp.name
    tmp.close()

    X_memmap = np.memmap(X_path, dtype=X.dtype, mode='w+', shape=X.shape)
    np.copyto(X_memmap, X)
    X_memmap.flush()
    del X_memmap

    try:
        yield {f"{name}_path": X_path, f"{name}_shape": X.shape, f"{name}_dtype": X.dtype}
    finally:
        if os.path.exists(X_path):
            os.remove(X_path)
