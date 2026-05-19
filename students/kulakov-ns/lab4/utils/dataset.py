from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
TARGET = "target"



def ensure_dataset(path: str = "data/iris.csv") -> Path:
    dataset_path = Path(path)
    if dataset_path.exists():
        return dataset_path

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True)
    frame = iris.frame.copy()
    frame["species"] = frame[TARGET].map(dict(enumerate(iris.target_names)))
    frame.to_csv(dataset_path, index=False)
    return dataset_path



def load_iris_dataset(
    path: str = "data/iris.csv",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    dataset_path = ensure_dataset(path)
    df = pd.read_csv(dataset_path)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    preprocessor = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    X_train_prep = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=FEATURES,
        index=X_train.index,
    ).reset_index(drop=True)
    X_test_prep = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=FEATURES,
        index=X_test.index,
    ).reset_index(drop=True)

    return (
        X_train_prep,
        X_test_prep,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        FEATURES,
    )
