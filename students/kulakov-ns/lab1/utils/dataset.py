from typing import Tuple, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def load_titanic(path: str = "data/train.csv", test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(path)

    df["CabinDeck"] = df["Cabin"].str[0]

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "CabinDeck"]
    target = "Survived"

    X = df[features].copy()
    y = df[target].copy()

    categorical_features = ["Pclass", "Sex", "Embarked", "CabinDeck"]
    numeric_features = ["Age", "SibSp", "Parch", "Fare"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    feature_types = {col: "cat" for col in categorical_features}
    feature_types.update({col: "num" for col in numeric_features})

    return X_train, X_test, y_train, y_test, feature_types