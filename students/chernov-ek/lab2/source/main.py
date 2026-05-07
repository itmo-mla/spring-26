import argparse
import time

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_archive, load_df, preprocess_df
from models import RandomForestClassifier as CustomRandomForestClassifier
from utils import grid_search_oob


def parse_arguments() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки для запуска обучения.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        argparse.Namespace: Пространство имён с параметрами запуска.

    Fallbacks:
        Если параметр не указан, используется режим обработки пропусков "none".
    """
    # Создаём парсер для настройки запуска из терминала.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-grid-search",
        default="no",
        choices=["no", "yes"],
        help="Подобрать оптимальные гипер-параметры: no или yes.",
    )
    parser.add_argument(
        "--use-custom-model",
        default="yes",
        choices=["no", "yes"],
        help="Использовать кастомную модель: no или yes.",
    )
    parser.add_argument(
        "--set-timer",
        default="no",
        choices=["no", "yes"],
        help="Замерить время обучения: no или yes.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Загружает данные, обучает случайный лес и выводит accuracy.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        None: Результат оценки выводится в консоль.

    Fallbacks:
        Ошибки загрузки данных и обучения передаются вызывающему коду.
    """
    # Разбираем параметры запуска перед обучением модели.
    arguments = parse_arguments()

    # Загружаем архив и читаем таблицу с датасетом.
    load_archive()
    df = load_df()

    # Предобрабатываем и делим выборки
    X_train, X_test, y_train, y_test = preprocess_df(df)

    if arguments.use_grid_search == "yes":
        # Определяем параметры для обучения
        param_grid = {
            "n_estimators": [20, 50, 100],
            "max_depth": [None, 5, 10],
            "max_features": ["sqrt", "log2"]
        }
        
        # Получаем лучшие параметры и обученную модель
        best_params, rf = grid_search_oob(
            X_train,
            y_train,
            param_grid,
            CustomRandomForestClassifier if arguments.use_custom_model == "yes" else RandomForestClassifier
        )

        print("Best params:", best_params)
    else:
        # Создаем модель случайного леса
        model = CustomRandomForestClassifier if arguments.use_custom_model == "yes" else RandomForestClassifier
        rf = model(
            n_estimators=50,
            max_depth=5,
            max_features="sqrt",
            random_state=42
        )

        if arguments.set_timer  == "yes":
            start_time = time.perf_counter()

        # Обучаем модель
        rf.fit(X_train, y_train)

        if arguments.set_timer == "yes":
            train_time = time.perf_counter() - start_time
            print(f"Training time: {train_time:.4f} sec")

    # Получаем предсказания
    y_predicted = rf.predict(X_test)

    # Выводим в консоль Accuracy и OOB score
    print(classification_report(y_test, y_predicted, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    
    if arguments.use_custom_model == "yes":
        assert isinstance(rf, CustomRandomForestClassifier)
        print("OOB score:", rf.oob_score(X_train, y_train))


if __name__ == "__main__":
    main()
