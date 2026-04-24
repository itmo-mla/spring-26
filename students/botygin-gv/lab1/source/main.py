import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

from data_loader import DataLoader
from tree import CustomDecisionTree
from pruning import TreePruner

DATA_PATH = "../dataset.csv"
TARGET_COLUMN = "Survived"
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
MIN_GAIN_THRESHOLD = 0.01
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42


def main():
    loader = DataLoader(DATA_PATH, TARGET_COLUMN, categorical_features=['Sex', 'Embarked', 'Pclass'],
                        numerical_features=['Age', 'SibSp', 'Parch', 'Fare'],
                        drop_columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    X_train, X_test, y_train, y_test, feature_names = loader.preprocess(
        test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество пропусков: {np.isnan(X_train).sum()}")

    print("Обучение решающего дерева...")
    tree = CustomDecisionTree(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_gain=MIN_GAIN_THRESHOLD
    )
    tree.fit(X_train, y_train, feature_names)

    pred_before = tree.predict(X_test)
    acc_before = accuracy_score(y_test, pred_before)
    f1_before = f1_score(y_test, pred_before)
    print(f"Точность до редукции: {acc_before:.4f}, f1 score: {f1_before:.4f}")

    print("Редукция дерева (Reduced Error Pruning)...")
    pruner = TreePruner(tree)
    pruner.prune(X_test, y_test)

    pred_after = tree.predict(X_test)
    acc_after = accuracy_score(y_test, pred_after)
    f1_after = f1_score(y_test, pred_after)
    print(f"Точность после редукции: {acc_after:.4f}, f1 score: {f1_after:.4f}")

    sklearn_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    sklearn_f1 = f1_score(y_test, sklearn_pred)
    print(f"Точность реализации из Sklearn: {sklearn_acc:.4f}, f1 score: {sklearn_f1:.4f}")


if __name__ == "__main__":
    main()
