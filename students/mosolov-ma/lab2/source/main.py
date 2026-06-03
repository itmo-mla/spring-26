import time
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance
from kagglehub import KaggleDatasetAdapter
import kagglehub


os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("random_forest")
logger.setLevel(logging.INFO)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
handler = logging.FileHandler(f"logs/random_forest_{timestamp}.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


class RandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None,
                 min_samples_split=2, random_state=None, oob_selection=True):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.oob_selection = oob_selection
        self.trees = []
        self.oob_indices_ = []
        self.tree_oob_scores_ = []
        self.selected_tree_indices_ = []
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        self.trees = []
        self.oob_indices_ = []
        self.tree_oob_scores_ = []
        all_trees = []
        all_oob_indices = []
        all_tree_oob_scores = []

        for i in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(boot_idx))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31))
            )
            tree.fit(X[boot_idx], y[boot_idx])

            tree_oob_score = np.nan
            if len(oob_idx) > 0:
                tree_oob_score = accuracy_score(y[oob_idx], tree.predict(X[oob_idx]))

            all_trees.append(tree)
            all_oob_indices.append(oob_idx)
            all_tree_oob_scores.append(tree_oob_score)

        if self.oob_selection:
            valid_scores = np.array([score for score in all_tree_oob_scores if not np.isnan(score)])
            if valid_scores.size > 0:
                threshold = float(valid_scores.mean())
                selected_indices = [
                    i for i, score in enumerate(all_tree_oob_scores)
                    if not np.isnan(score) and score >= threshold
                ]
                if len(selected_indices) == 0:
                    selected_indices = [
                        i for i, score in enumerate(all_tree_oob_scores)
                        if not np.isnan(score)
                    ]
            else:
                selected_indices = list(range(len(all_trees)))
        else:
            selected_indices = list(range(len(all_trees)))

        if len(selected_indices) == 0:
            selected_indices = list(range(len(all_trees)))

        self.selected_tree_indices_ = selected_indices
        self.trees = [all_trees[i] for i in selected_indices]
        self.oob_indices_ = [all_oob_indices[i] for i in selected_indices]
        self.tree_oob_scores_ = [all_tree_oob_scores[i] for i in selected_indices]
        return self

    def oob_score(self):
        n_samples = self.X_train.shape[0]
        oob_sum = np.zeros(n_samples)
        oob_count = np.zeros(n_samples)

        for tree, oob_idx in zip(self.trees, self.oob_indices_):
            if len(oob_idx) > 0:
                preds = tree.predict(self.X_train[oob_idx])
                oob_sum[oob_idx] += preds
                oob_count[oob_idx] += 1

        valid = oob_count > 0
        if valid.sum() == 0:
            return 0.0
        oob_pred = np.round(oob_sum[valid] / oob_count[valid]).astype(int)
        return accuracy_score(self.y_train[valid], oob_pred)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.round(np.mean(predictions, axis=1)).astype(int)

    def feature_importance_oob(self):
        n_samples, n_features = self.X_train.shape
        oob_sum = np.zeros(n_samples)
        oob_count = np.zeros(n_samples)

        for tree, oob_idx in zip(self.trees, self.oob_indices_):
            if len(oob_idx) > 0:
                oob_sum[oob_idx] += tree.predict(self.X_train[oob_idx])
                oob_count[oob_idx] += 1

        valid = oob_count > 0
        if valid.sum() == 0:
            return np.zeros(n_features)

        baseline_pred = np.round(oob_sum[valid] / oob_count[valid]).astype(int)
        baseline_acc = accuracy_score(self.y_train[valid], baseline_pred)

        importances = np.zeros(n_features)
        for j in range(n_features):
            X_perm = self.X_train.copy()
            rng = np.random.default_rng(42)
            rng.shuffle(X_perm[:, j])

            perm_sum = np.zeros(n_samples)
            perm_count = np.zeros(n_samples)
            for tree, oob_idx in zip(self.trees, self.oob_indices_):
                if len(oob_idx) > 0:
                    perm_sum[oob_idx] += tree.predict(X_perm[oob_idx])
                    perm_count[oob_idx] += 1

            perm_pred = np.round(perm_sum[valid] / perm_count[valid]).astype(int)
            perm_acc = accuracy_score(self.y_train[valid], perm_pred)
            importances[j] = baseline_acc - perm_acc

        return importances


def grid_search_oob(X, y, param_grid, random_state=42):
    grid = ParameterGrid(param_grid)
    best_score = -1
    best_params = None

    for params in grid:
        model = RandomForest(random_state=random_state, **params)
        model.fit(X, y)
        score = model.oob_score()
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


def main():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "nareshbhat/health-care-data-set-on-heart-attack-possibility",
        "heart.csv"
    )

    X = df.drop(columns=['target']).values
    y = df['target'].values
    feature_names = df.drop(columns=['target']).columns

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5]
    }

    logger.info("Подбор гиперпараметров по OOB...")
    best_params, best_oob = grid_search_oob(X, y, param_grid, random_state=42)
    logger.info(f"Лучшие параметры: {best_params}")
    logger.info(f"Лучший OOB score: {best_oob:.4f}")

    start = time.time()
    my_rf = RandomForest(random_state=42, **best_params)
    my_rf.fit(X, y)
    my_time = time.time() - start
    my_oob = my_rf.oob_score()
    my_acc = accuracy_score(y, my_rf.predict(X))
    my_imp = my_rf.feature_importance_oob()

    logger.info("Custom:")
    logger.info(f"  Время обучения: {my_time:.4f} сек")
    logger.info(f"  OOB score: {my_oob:.4f}")
    logger.info(f"  Accuracy: {my_acc:.4f}")
    logger.info("  Feature importances (OOB^j):")
    for name, imp in zip(feature_names, my_imp):
        logger.info(f"    {name}: {imp:.4f}")

    start = time.time()
    skl_rf = SklearnRF(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params.get('max_depth'),
        max_features=best_params['max_features'],
        min_samples_split=best_params.get('min_samples_split'),
        oob_score=True,
        random_state=42
    )
    skl_rf.fit(X, y)
    skl_time = time.time() - start
    skl_acc = skl_rf.score(X, y)
    skl_oob = skl_rf.oob_score_
    skl_imp = skl_rf.feature_importances_

    perm_result = permutation_importance(skl_rf, X, y, n_repeats=10, random_state=42)
    skl_perm_imp = perm_result.importances_mean

    logger.info("sklearn реализация:")
    logger.info(f"  Время обучения: {skl_time:.4f} сек")
    logger.info(f"  OOB score: {skl_oob:.4f}")
    logger.info(f"  Accuracy: {skl_acc:.4f}")
    logger.info("  Feature importances (impurity):")
    for name, imp in zip(feature_names, skl_imp):
        logger.info(f"    {name}: {imp:.4f}")

    logger.info(f"sklearn permutation_importance:")
    for name, imp in zip(feature_names, skl_perm_imp):
        logger.info(f"    {name}: {imp:.4f}")

    logger.info("")
    logger.info("Сравнение:")
    header = f"{'Метрика':<30} {'custom':<20} {'sklearn':<20}"
    logger.info(header)
    logger.info("-" * len(header))
    logger.info(f"{'Accuracy':<30} {my_acc:<20.4f} {skl_acc:<20.4f}")
    logger.info(f"{'OOB score':<30} {my_oob:<20.4f} {skl_oob:<20.4f}")
    logger.info(f"{'Время обучения (сек)':<30} {my_time:<20.4f} {skl_time:<20.4f}")

    logger.info("")
    header = f"{'Признак':<15} {'OOB^j (custom)':<20} {'sklearn permutation':<20}"
    logger.info("Сравнение feature importances:")
    logger.info(header)
    logger.info("-" * len(header))
    for name, v1, v2, v3 in zip(feature_names, my_imp, skl_imp, skl_perm_imp):
        logger.info(f"{name:<15} {v1:<20.4f} {v3:<20.4f}")


if __name__ == '__main__':
    main()
