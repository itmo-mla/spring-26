from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def build_sklearn_pipeline(
    task: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any],
) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_features)
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    tree_params = _filter_tree_params(model_params)
    if task == "classification":
        tree = DecisionTreeClassifier(**tree_params)
    elif task == "regression":
        tree = DecisionTreeRegressor(**tree_params)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", tree)])


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return [str(name) for name in preprocessor.get_feature_names_out()]


def _filter_tree_params(model_params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "min_impurity_decrease",
        "ccp_alpha",
        "random_state",
    }
    return {key: value for key, value in model_params.items() if key in allowed}
