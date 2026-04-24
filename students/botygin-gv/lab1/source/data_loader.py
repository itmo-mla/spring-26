import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataLoader:
    def __init__(self, path: str, target_col: str, categorical_features: list, numerical_features: list,
                 drop_columns: list):
        self.df = pd.read_csv(path, encoding="utf-8", delimiter=",")
        self.target_col = target_col
        self.feature_names = []
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.drop_columns = drop_columns

    def preprocess(self, test_size=0.2, random_state=42):
        cols_to_drop = [col for col in self.drop_columns if col in self.df.columns]
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)

        X = self.df.drop(columns=[self.target_col]).copy()
        y = self.df[self.target_col].copy()

        X_cat = X[self.categorical_features].copy()
        X_num = X[self.numerical_features].copy()

        X_cat = X_cat.fillna('Unknown')
        X_cat_encoded = self.encoder.fit_transform(X_cat)

        encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_features)

        X_num = X_num.astype(float)
        X_encoded = np.hstack([X_cat_encoded, X_num.values])

        self.feature_names = list(encoded_feature_names) + self.numerical_features

        X = pd.DataFrame(X_encoded, columns=self.feature_names, index=X.index)
        X = X.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train.values, X_test.values, y_train.values, y_test.values, self.feature_names
