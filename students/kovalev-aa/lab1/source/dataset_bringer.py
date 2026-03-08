import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class AdultDataset:
    MISSING_RATE = 0.1
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TARGET = "class"

    # Основные признаки (категориальные и числовые)
    USEFUL_FEATURES = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country"
    ]

    @classmethod
    def get_dataset_split(cls):
        df = cls.__load_and_clean()
        df = cls.__impute_missing(df)
        df = cls.__encode_target(df)  # <- Сначала целевой признак
        df = cls.__encode_categorical(df)  # <- Потом one-hot для признаков

        y = df[cls.TARGET]
        X = df.drop(cls.TARGET, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cls.TEST_SIZE,
            random_state=cls.RANDOM_STATE
        )

        return x_train, x_test, y_train, y_test

    @classmethod
    def __load_and_clean(cls):
        data = fetch_openml("adult", version=2, as_frame=True)
        df = data.frame.copy()
        df = df[cls.USEFUL_FEATURES + [cls.TARGET]]

        # Очистка строковых признаков
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.lower().str.strip()

        # Искусственные пропуски
        rng = np.random.default_rng(cls.RANDOM_STATE)
        for col in df.columns:
            if col != cls.TARGET and df[col].notna().sum() > 0:
                mask = rng.random(df.shape[0]) < cls.MISSING_RATE
                df.loc[mask, col] = np.nan

        return df

    @staticmethod
    def __impute_missing(df):
        rng = np.random.default_rng(AdultDataset.RANDOM_STATE)
        df_filled = df.copy()

        for col in df.columns:
            if col == AdultDataset.TARGET:
                continue
            if df[col].dtype in ['int64', 'float64']:
                df_filled[col] = df[col].fillna(df[col].mean())
            else:
                counts = df[col].value_counts(normalize=True, dropna=True)
                missing_idx = df[col].isna()
                df_filled.loc[missing_idx, col] = rng.choice(
                    counts.index,
                    size=missing_idx.sum(),
                    p=counts.values
                )
        return df_filled

    @classmethod
    def __encode_categorical(cls, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        cat_cols = [c for c in cat_cols if c != cls.TARGET]

        df_cat = pd.get_dummies(df_encoded[cat_cols], drop_first=True)
        num_cols = [c for c in df_encoded.columns if c not in cat_cols + [cls.TARGET]]
        df_num = df_encoded[num_cols].copy()

        df_final = pd.concat([df_num, df_cat, df_encoded[cls.TARGET]], axis=1)
        return df_final.apply(pd.to_numeric)

    @classmethod
    def __encode_target(cls, df):
        df_encoded = df.copy()
        le = LabelEncoder()
        df_encoded[cls.TARGET] = le.fit_transform(df_encoded[cls.TARGET])
        return df_encoded
