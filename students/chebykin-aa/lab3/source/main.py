import logging
import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier as SklearnGB
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate as sk_cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import GradientBoostingClassifier as CustomGB
from utils import load_data, cross_validate, save_cv_results

def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str,
    params: dict,
    cv: int,
    seed: int,
):
    # Кросс-валидация Custom GB
    t0 = time.perf_counter()
    custom_scores = cross_validate(X, y, params, cv=cv, random_state=seed)
    custom_time = time.perf_counter() - t0

    save_cv_results(custom_scores, f"Custom GB — {cv}-fold CV",
                     os.path.join(save_path, "custom_gb_cv.txt"))
    logging.info(f"Время обучения Custom GB ({cv}-fold): {custom_time:.3f}s")

    # Кросс-валидация sklearn GB с теми же параметрами
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    sk_gb = SklearnGB(**params, random_state=seed)
    sk_raw = sk_cross_validate(sk_gb, X, y, cv=skf,
                               scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"])
    sk_time = time.perf_counter() - t0

    # Приводим ключи к единому формату
    sk_scores = {
        "accuracy": sk_raw["test_accuracy"],
        "precision": sk_raw["test_precision_weighted"],
        "recall": sk_raw["test_recall_weighted"],
        "f1": sk_raw["test_f1_weighted"],
    }
    save_cv_results(sk_scores, f"sklearn GB — {cv}-fold CV",
                     os.path.join(save_path, "sklearn_gb_cv.txt"))
    logging.info(f"Время обучения sklearn GB ({cv}-fold): {sk_time:.3f}s")

    # Сравнение времени обучения
    with open(os.path.join(save_path, "timing.txt"), "w", encoding="utf-8") as f:
        f.write(f"Время обучения Custom GB  ({cv}-fold CV): {custom_time:.3f}s\n")
        f.write(f"Время обучения sklearn GB ({cv}-fold CV): {sk_time:.3f}s\n")
        f.write(f"Соотношение (custom/sklearn): {custom_time / sk_time:.1f}x\n")

    # Матрицы ошибок: обучаем на всех данных
    custom_gb = CustomGB(**params, random_state=seed)
    custom_gb.fit(X, y)
    y_pred_custom = custom_gb.predict(X)

    sk_gb_full = SklearnGB(**params, random_state=seed)
    sk_gb_full.fit(X, y)
    y_pred_sklearn = sk_gb_full.predict(X)

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, y_pred) in zip(axes, [
        ("Custom GB", y_pred_custom),
        ("sklearn GB", y_pred_sklearn),
    ]):
        ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot(ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrices.png"), dpi=300)
    plt.close()

def main():
    SEED = 42
    CV = 5
    PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
    }

    # Создаём директорию для результатов
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path, exist_ok=True)

    # Настраиваем логирование
    logging.basicConfig(
        filename=os.path.join(results_path, "main.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Загрузка данных
    X, y, _, _ = load_data()

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    missing_mask = np.isnan(X)
    logging.info(f"Samples: {X.shape[0]}")
    logging.info(f"Классы: выжил=0 ({(y == 0).sum()}), умер=1 ({(y == 1).sum()})")
    logging.info(f"Пропуски: {int(missing_mask.sum())} ({missing_mask.mean() * 100:.1f} %)")

    run_pipeline(X_imp, y, save_path=results_path, params=PARAMS, cv=CV, seed=SEED)

if __name__ == "__main__":
    main()
