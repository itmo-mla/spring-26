import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(
    csv_path="data/breast-cancer.csv",
    test_size=0.2,
    random_state=42
):

    df = pd.read_csv(csv_path)

    drop_cols = []

    if "id" in df.columns:
        drop_cols.append("id")

    if "Unnamed: 32" in df.columns:
        drop_cols.append("Unnamed: 32")

    df.drop(columns=drop_cols, inplace=True)


    df["diagnosis"] = df["diagnosis"].map({
        "M": 1,
        "B": 0
    })


    y = df["diagnosis"].values
    X = df.drop(columns=["diagnosis"])

    feature_names = X.columns.tolist()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    )
