import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
import numpy as np


class BaseModel:
    def __init__(self, model):
        self.model = model


    def fit(self, X, y) -> None:
        self.model.fit(X, y)


    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class TrainedModelInfo:
    model: BaseModel
    feature_idx: np.ndarray
    train_idx: np.ndarray


class DecisionTree(BaseModel):
    def __init__(self, **kwargs):
        model = DecisionTreeClassifier(**kwargs)
        super().__init__(model)

    def fit(self, X, y) -> None:
        self.model.fit(X, y)


class RandomForest:
    def __init__(self, base_model_class: BaseModel, n_algorithms: int = 100, **kwargs):
        self.base_model_class = base_model_class
        self.base_model_params = kwargs
        self.n_algorithms = n_algorithms


    def __get_random_features_idx(self, X: np.ndarray) -> np.ndarray:
        n_features = int(np.sqrt(X.shape[1]))
        features_to_select = np.random.choice(X.shape[1], size=n_features, replace=False)
        return features_to_select


    def __get_bootstrap_idx(self, n_samples: int) -> np.ndarray:
        return np.random.choice(n_samples, size=n_samples, replace=True)


    def __fit_single_model(self, model_instance, X, y) -> tuple[TrainedModelInfo, list[float]]:
        feature_idx = self.__get_random_features_idx(X)
        train_idx = self.__get_bootstrap_idx(X.shape[0])

        X_train = X[train_idx][:, feature_idx]
        y_train = y[train_idx]

        model_instance.fit(X_train, y_train)

        train_acc = np.mean(model_instance.predict(X_train) == y_train)

        val_mask = np.ones(X.shape[0], dtype=bool)
        val_mask[np.unique(train_idx)] = False

        if np.any(val_mask):
            X_val = X[val_mask][:, feature_idx]
            y_val = y[val_mask]
            val_acc = np.mean(model_instance.predict(X_val) == y_val)
        else:
            val_acc = 0.0

        info = TrainedModelInfo(
            model=model_instance,
            feature_idx=feature_idx,
            train_idx=train_idx,
        )
        return info, [1 - train_acc, 1 - val_acc]


    def fit(self, X, y) -> None:
        self.trained_models = []
        eps1 = 0.3
        eps2 = 0.4

        while len(self.trained_models) < self.n_algorithms:
            base_model_instance = self.base_model_class(**self.base_model_params)
            trained_model_info, errors = self.__fit_single_model(base_model_instance, X, y)
            
            if errors[0] <= eps1 and errors[1] <= eps2:
                self.trained_models.append(trained_model_info)
            else:
                print(f"{errors=}")


    def compute_train_score(self, X, y) -> float:
        vote_sum = np.zeros(X.shape[0], dtype=float)
        vote_count = np.zeros(X.shape[0], dtype=int)

        for model_info in self.trained_models:
            train_mask = np.zeros(X.shape[0], dtype=bool)
            train_mask[np.unique(model_info.train_idx)] = True

            if not np.any(train_mask):
                continue

            X_train = X[train_mask][:, model_info.feature_idx]
            preds = model_info.model.predict(X_train)
            vote_sum[train_mask] += preds
            vote_count[train_mask] += 1
        
        valid_mask = vote_count > 0
        final_preds = np.round(vote_sum[valid_mask] / vote_count[valid_mask])
        return np.mean(final_preds == y[valid_mask])
    
    
    def compute_oob_score(self, X, y) -> float:
        vote_sum = np.zeros(X.shape[0], dtype=float)
        vote_count = np.zeros(X.shape[0], dtype=int)

        for model_info in self.trained_models:
            oob_mask = np.ones(X.shape[0], dtype=bool)
            oob_mask[np.unique(model_info.train_idx)] = False

            if not np.any(oob_mask):
                continue

            X_oob = X[oob_mask][:, model_info.feature_idx]
            preds = model_info.model.predict(X_oob)
            vote_sum[oob_mask] += preds
            vote_count[oob_mask] += 1
        
        valid_mask = vote_count > 0
        final_preds = np.round(vote_sum[valid_mask] / vote_count[valid_mask])
        return np.mean(final_preds == y[valid_mask])
    
    
    def get_feature_importance(self, X, y) -> np.ndarray:
        oob = self.compute_oob_score(X, y)

        feature_importances = []
        for i in range(X.shape[1]):
            X_modified = X.copy()
            shuffled_feature = np.random.shuffle(X_modified[:, i])
            X_modified[:, i] = shuffled_feature
            shuffled_oob = self.compute_oob_score(X_modified, y)

            feature_importances.append((shuffled_oob - oob) / oob * 100)
        
        return np.array(feature_importances)


    def __get_raw_predictions(self, X: np.ndarray) -> np.ndarray:
        return np.array([model_info.model.predict(X[:, model_info.feature_idx]) for model_info in self.trained_models])


    def predict(self, X) -> np.ndarray:
        return np.mean(self.__get_raw_predictions(X), axis=0)
