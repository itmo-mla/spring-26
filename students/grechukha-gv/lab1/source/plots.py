"""Визуализация метрик и матриц ошибок"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from tree import DecisionTreeID3


def plot_metrics_bar(results_test, path):
    models = list(results_test.keys())
    acc = [results_test[m]["accuracy"] for m in models]
    prec = [results_test[m]["precision"] for m in models]
    rec = [results_test[m]["recall"] for m in models]
    f1 = [results_test[m]["f1"] for m in models]

    x = np.arange(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * w, acc, w, label="Accuracy")
    ax.bar(x - 0.5 * w, prec, w, label="Precision")
    ax.bar(x + 0.5 * w, rec, w, label="Recall")
    ax.bar(x + 1.5 * w, f1, w, label="F1-score")
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение качества классификации на тестовой выборке")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_triple(y_test, pred_raw, pred_pruned, pred_sklearn, path):
    matrices = [
        confusion_matrix(y_test, pred_raw),
        confusion_matrix(y_test, pred_pruned),
        confusion_matrix(y_test, pred_sklearn),
    ]
    titles = ["ID3 до редукции", "ID3 после редукции", "sklearn (эталон)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, cm, title in zip(axes, matrices, titles):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Количество объектов")
        thr = cm.max() / 2 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thr else "black",
                )
    fig.suptitle("Матрицы ошибок (тест)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_max_depth(
    X_train, y_train, X_test, y_test, feature_names, path, max_depth=12
):
    """Train/test accuracy при ограничении глубины дерева (отдельное дерево на каждую глубину)"""
    depths = list(range(1, max_depth + 1))
    train_acc = []
    test_acc = []
    for d in depths:
        clf = DecisionTreeID3(max_depth=d, min_samples_split=2)
        clf.fit(X_train, y_train, feature_names)
        pred_tr = clf.predict(X_train)
        pred_te = clf.predict(X_test)
        train_acc.append(np.mean(pred_tr == y_train))
        test_acc.append(np.mean(pred_te == y_test))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, train_acc, marker="o", label="Train accuracy")
    ax.plot(depths, test_acc, marker="s", label="Test accuracy")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Зависимость точности от ограничения глубины (ID3)")
    ax.legend()
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(counts_dict, path):
    """Нормированные частоты сплитов по признакам"""
    if not counts_dict:
        return
    feats = sorted(counts_dict.keys(), key=lambda k: counts_dict[k], reverse=True)
    vals = np.array([counts_dict[f] for f in feats], dtype=float)
    vals = vals / vals.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(feats[::-1], vals[::-1], color="steelblue")
    ax.set_xlabel("Доля сплитов по признаку")
    ax.set_title("Важность признаков (число сплитов в дереве до редукции)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
