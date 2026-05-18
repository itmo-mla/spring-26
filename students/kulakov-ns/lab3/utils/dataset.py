from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "CabinDeck"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Survived"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def load_titanic(
    path: str = "data/train.csv",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    df = pd.read_csv(path)
    df["CabinDeck"] = df["Cabin"].str[0]

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    X_train_prep = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=preprocessor.get_feature_names_out(),
        index=X_train.index,
    ).reset_index(drop=True)
    X_test_prep = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out(),
        index=X_test.index,
    ).reset_index(drop=True)

    return (
        X_train_prep,
        X_test_prep,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        FEATURES,
    )
