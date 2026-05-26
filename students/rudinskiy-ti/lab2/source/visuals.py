import pandas as pd
import numpy as np
from RandomForestClassifier import RandomForestClassifier
import matplotlib.pyplot as plt 

def train_loss_plot(model:RandomForestClassifier, X_train, y_true, figsize: tuple = (8, 5)):
    K = len(model.classifiers)
    loss_arr = []
    for i in range(K):
        loss_arr.append(model._partial_predict(X_train, i+1, y_true))
    plt.figure(figsize=figsize)
    plt.plot(range(1, K + 1), loss_arr)
    plt.xlabel("Количество деревьев в ансамбле", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Динамика лосса на обучающей выборке", fontsize=13, pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def test_loss_plot(model:RandomForestClassifier, X_test, y_true, figsize: tuple = (8, 5)):
    K = len(model.classifiers)
    loss_arr = []
    for i in range(K):
        loss_arr.append(model._partial_predict(X_test, i+1, y_true))
    plt.figure(figsize=figsize)
    plt.plot(range(1, K + 1), loss_arr)
    plt.xlabel("Количество деревьев в ансамбле", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Динамика лосса на обучающей выборке", fontsize=13, pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    