from pathlib import Path
import pandas as pd


ROOT_PATH = Path(__file__).parent.parent
TRAIN_DATASET_PATH = ROOT_PATH / "data" / "flight_delays_train.csv"


def load_df():
    """
    Загружает исходный датасет задержек рейсов.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        DataFrame: Таблица с признаками и целевым столбцом.

    Fallbacks:
        Ошибка чтения файла передается вызывающему коду.
    """
    # Читаем данные из локального csv-файла.
    return pd.read_csv(TRAIN_DATASET_PATH)


def preprocess_data(df):
    """
    Кодирует признаки и выделяет целевую переменную.

    Parameters:
        df (DataFrame): Исходная таблица с данными. По умолчанию: None.

    Returns:
        tuple: Матрица признаков, целевой вектор и имена признаков.

    Fallbacks:
        Некорректные значения в целевом столбце превращаются в NaN и будут обнаружены моделью.
    """
    # Отделяем целевой столбец от признаков.
    y = df.pop("dep_delayed_15min").map({"Y": 1, "N": 0}).to_numpy()

    # Кодируем категориальные признаки прямым one-hot кодированием.
    X_df = pd.get_dummies(df, dtype="float32")
    X = X_df.to_numpy(dtype="float32")

    # Возвращаем подготовленные данные для эксперимента.
    return X, y


def data_pipeline():
    """
    Выполняет полный цикл подготовки данных.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        tuple: Матрица признаков, целевой вектор и имена признаков.

    Fallbacks:
        Ошибки загрузки и подготовки данных передаются вызывающему коду.
    """
    # Загружаем и подготавливаем данные одним вызовом.
    df = load_df()
    return preprocess_data(df)
