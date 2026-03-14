import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str | float | int = "sqrt",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Заполняются после fit()
        self.estimators_: list[DecisionTreeClassifier] = []
        self._oob_masks: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.oob_score_: float | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, _ = X.shape
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self._oob_masks = []

        # Будем запоминать, какие объекты видела каждая модель
        oob_votes = np.zeros((n_samples, len(self.classes_)), dtype=float)
        oob_counts = np.zeros(n_samples, dtype=int)

        for _ in range(self.n_estimators):
            # Получаем бутстрапированную выборку
            boot_idx = self.rng.randint(0, n_samples, size=n_samples)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False

            X_boot, y_boot = X[boot_idx], y[boot_idx]

            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.rng.randint(0, 2**31),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            self._oob_masks.append(oob_mask)

            # Накапливаем OOB-голоса
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) > 0:
                proba = tree.predict_proba(X[oob_idx])
                # Выравниваем столбцы, т.к. дерево может не знать все классы
                for k, cls in enumerate(self.classes_):
                    if cls in tree.classes_:
                        tree_k = np.where(tree.classes_ == cls)[0][0]
                        oob_votes[oob_idx, k] += proba[:, tree_k]
                oob_counts[oob_idx] += 1

        # Считаем OOB-точность
        voted = oob_counts > 0
        if voted.sum() > 0:
            oob_preds = self.classes_[np.argmax(oob_votes[voted], axis=1)]
            self.oob_score_ = float(np.mean(oob_preds == y[voted]))
        else:
            self.oob_score_ = 0.0

        # Получаем важность признаков через OOB-перестановку
        self.feature_importances_ = self._compute_feature_importances(X, y)

    def predict(self, X: np.ndarray):
        # Голосование по всем деревьям
        votes = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for tree in self.estimators_:
            proba = tree.predict_proba(X)
            for k, cls in enumerate(self.classes_):
                if cls in tree.classes_:
                    tree_k = np.where(tree.classes_ == cls)[0][0]
                    votes[:, k] += proba[:, tree_k]
                    
        return self.classes_[np.argmax(votes, axis=1)]

    def _compute_feature_importances(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        count = 0

        # Важность признака = среднее падение OOB-точности при перестановке его значений
        for tree, oob_mask in zip(self.estimators_, self._oob_masks):
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) < 2:
                continue

            X_oob = X[oob_idx]
            y_oob = y[oob_idx]
            baseline_acc = np.mean(tree.predict(X_oob) == y_oob)

            for j in range(n_features):
                X_perm = X_oob.copy()
                X_perm[:, j] = self.rng.permutation(X_perm[:, j])
                perm_acc = np.mean(tree.predict(X_perm) == y_oob)
                importances[j] += baseline_acc - perm_acc

            count += 1

        if count > 0:
            importances /= count

        # Убираем признаки с отрицательной важностью
        importances = np.maximum(importances, 0.0)

        # Нормируем, чтобы сумма была равна 1
        total = importances.sum()
        if total > 0:
            importances /= total

        return importances
