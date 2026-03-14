import logging
import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import RandomForestClassifier as CustomRF
from utils import load_data, evaluate, oob_grid_search


def run_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    save_path: str,
    param_grid: dict,
    seed: int,
):
    # Подбор гиперпараметров по OOB
    best_params, best_oob = oob_grid_search(X_train, y_train, param_grid, random_state=seed)
    logging.info(f"Лучшие параметры: {best_params}  OOB={best_oob:.4f}")

    with open(os.path.join(save_path, "grid_search.txt"), "w", encoding="utf-8") as f:
        f.write(f"Лучшие гиперпараметры:\n{best_params}\n")
        f.write(f"OOB accuracy: {best_oob:.4f}\n")

    # Обучение финального Custom RF
    t0 = time.perf_counter()
    custom_rf = CustomRF(**best_params, random_state=seed)
    custom_rf.fit(X_train, y_train)
    custom_train_time = time.perf_counter() - t0

    y_pred_custom = custom_rf.predict(X_test)
    evaluate(y_test, y_pred_custom, "Custom RF – Test", save_path=os.path.join(save_path, "custom_rf_test.txt"))
    logging.info(f"Время обучения Custom RF: {custom_train_time:.3f}s")
    logging.info(f"OOB-точность Custom RF: {custom_rf.oob_score_:.4f}")

    # Важность признаков через OOB-перестановку
    importances = custom_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    with open(os.path.join(save_path, "feature_importances.txt"), "w", encoding="utf-8") as f:
        f.write("Важность признаков (OOB-перестановка):\n")
        for rank, idx in enumerate(sorted_idx, 1):
            f.write(f"  {rank:2d}. {feature_names[idx]:<40s} {importances[idx]:.4f}\n")

    # График топ-15 признаков
    top_n = min(15, len(feature_names))
    top_idx = sorted_idx[:top_n]
    _, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[top_idx])
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha="right")
    ax.set_title("Важность признаков (OOB-перестановка) — топ 15")
    ax.set_ylabel("Среднее падение точности")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "feature_importances.png"), dpi=300)
    plt.close()

    # Эталонная реализация sklearn RF с теми же параметрами
    t0 = time.perf_counter()
    sk_rf = SklearnRF(**best_params, oob_score=True, random_state=seed)
    sk_rf.fit(X_train, y_train)
    sk_train_time = time.perf_counter() - t0

    y_pred_sklearn = sk_rf.predict(X_test)
    evaluate(y_test, y_pred_sklearn, "sklearn RF – Test", save_path=os.path.join(save_path, "sklearn_rf_test.txt"))
    logging.info(f"Время обучения sklearn RF: {sk_train_time:.3f}s")
    logging.info(f"OOB-точность sklearn RF: {sk_rf.oob_score_:.4f}")

    # Сравнение времени обучения
    with open(os.path.join(save_path, "timing.txt"), "w", encoding="utf-8") as f:
        f.write(f"Время обучения Custom RF : {custom_train_time:.3f}s\n")
        f.write(f"Время обучения sklearn RF: {sk_train_time:.3f}s\n")
        f.write(f"Соотношение (custom/sklearn): {custom_train_time / sk_train_time:.1f}x\n")

    # Матрицы ошибок
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, y_pred) in zip(axes, [
        ("Custom RF", y_pred_custom),
        ("sklearn RF", y_pred_sklearn),
    ]):
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrices.png"), dpi=300)
    plt.close()


def main():
    SEED = 42
    PARAM_GRID = {
        "n_estimators": [50, 100, 200],
        "max_features": ["sqrt", "log2"],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    # Создаём директорию для результатов
    results_path = os.path.join(os.getcwd(), "students", "chebykin-aa", "lab2", "results")
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
    X, y, feature_names, _ = load_data()

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    missing_mask = np.isnan(X)
    logging.info(f"Samples: {X.shape[0]}")
    logging.info(f"Классы: выжил=0 ({(y == 0).sum()}), умер=1 ({(y == 1).sum()})")
    logging.info(f"Пропуски: {int(missing_mask.sum())} ({missing_mask.mean() * 100:.1f} %)")

    # Разбиваем данные: 80%/20%
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.20, random_state=SEED, stratify=y
    )
    logging.info(f"Размер тренировочной выборки: {len(y_train)}")
    logging.info(f"Размер тестовой выборки: {len(y_test)}")

    run_pipeline(X_train, X_test, y_train, y_test, feature_names,
                 save_path=results_path, param_grid=PARAM_GRID, seed=SEED)


if __name__ == "__main__":
    main()
