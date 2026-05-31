from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Базовые пути проекта и архива с датасетом.
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"
ARCHIVE_NAME = "car-evaluation-data-set.zip"
DATASET_FILE_NAME = "car_evaluation.csv"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/elikplim/car-evaluation-data-set"
)


def load_archive() -> None:
    """
    Загружает архив с датасетом, если он отсутствует локально.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        None: Архив сохраняется в директорию data.

    Fallbacks:
        Если архив уже существует, загрузка пропускается.
    """
    # Создаем директорию с данными перед сохранением архива.
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_file = DATA_PATH / ARCHIVE_NAME

    # Не загружаем файл повторно, если он уже есть на диске.
    if output_file.exists():
        return

    # Получаем архив потоково, чтобы не держать весь файл в памяти.
    response = requests.get(DATASET_URL, stream=True, allow_redirects=True, timeout=60)
    response.raise_for_status()

    # Сохраняем архив порциями фиксированного размера.
    with output_file.open("wb") as archive_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                archive_file.write(chunk)


def load_df() -> pd.DataFrame:
    """
    Читает CSV-файл с датасетом из локального ZIP-архива.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        pandas.DataFrame: Таблица с признаками и целевой переменной.

    Fallbacks:
        Если архив отсутствует или поврежден, исключение передается вызывающему коду.
    """
    # Открываем архив и читаем CSV без промежуточной распаковки на диск.
    archive_path = DATA_PATH / ARCHIVE_NAME
    columns = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class"
    ]
    with zipfile.ZipFile(archive_path, "r") as archive:
        with archive.open(DATASET_FILE_NAME) as dataset_file:
            dataframe = pd.read_csv(dataset_file, header=None, names=columns)

    return dataframe


def preprocess_df(df: pd.DataFrame, random_state=None):
    """
    Предобработка DataFrame и разбиение на train/test выборки.

    Parameters:
        df (pd.DataFrame): Исходный DataFrame.
        random_state (int | None): Seed генератора случайных чисел.
            По умолчанию: None.

    Returns:
        tuple: Обучающая и тестовая выборки:
            X_train, X_test, y_train, y_test.

    Fallbacks:
        Если random_state не задан, используется
        случайная инициализация.
    """
    # Разделение признаков и целевой переменной
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Кодирование категориальных признаков
    X = X.apply(LabelEncoder().fit_transform)

    # Кодирование целевой переменной
    y = LabelEncoder().fit_transform(y)

    # Разбиение на обучающую и тестовую выборки
    return train_test_split(
        np.array(X),
        np.array(y),
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )


def main():
    load_archive()
    df = load_df()
    print(df.info())


if __name__ == "__main__":
    main()
