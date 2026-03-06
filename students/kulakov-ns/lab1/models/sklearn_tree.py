import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def get_sklearn_tree(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    categorical_features = ["Pclass", "Sex", "Embarked", "CabinDeck"]
    numeric_features = ["Age", "SibSp", "Parch", "Fare"]


    sklearn_pipeline = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), numeric_features),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_features,
                        ),
                    ]
                ),
            ),
            ("model", DecisionTreeClassifier(criterion="gini", random_state=42)),
        ]
    )
    return sklearn_pipeline.fit(X_train, y_train)