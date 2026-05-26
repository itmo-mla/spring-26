import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import math

class RandomForestClassifier:
    def __init__(self, l_sample:int, n_feat:int, n_estimators:int=50, tol_train:float=1e-2, tol_valid:float=1e-2, k_feature:int=None, **base_classifier_params):

        self.n_estimators = n_estimators
        self.l_sample = l_sample
        self.n_feat = n_feat
        self.tol_train = tol_train
        self.tol_valid = tol_valid
        k = k_feature
        if k_feature is None:
            k = math.floor(n_feat**0.5)
        self.base_classifier = DecisionTreeClassifier(
            max_features=k,
            **base_classifier_params
        )

        self.classifiers = []
        self.columns = []
        self.bacth_indexes = []

    def _error_rate(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

    def fit(self, X_train:pd.DataFrame, y_train:pd.Series):
        while len(self.classifiers) < self.n_estimators:
            model = clone(self.base_classifier)
            batch = X_train.sample(n=self.l_sample, axis=0, replace=True)
            batch_ind = batch.index
            batch_y = y_train.loc[batch.index]
            batch = batch.sample(n=self.n_feat, axis=1, replace=False)
            model.fit(batch, batch_y)

            y_pred_train = model.predict(batch)
            err1 = self._error_rate(batch_y,y_pred_train)

            ind_valid = X_train.index.difference(batch_ind)
            batch_valid = X_train.loc[ind_valid][batch.columns]
            batch_y_valid = y_train.loc[ind_valid]
            y_pred_valid = model.predict(batch_valid)
            err2 = self._error_rate(batch_y_valid, y_pred_valid)
            if err1 < self.tol_train and err2 < self.tol_valid:
                self.classifiers.append(model)
                self.columns.append(batch.columns)
                self.bacth_indexes.append(np.array(batch_ind))

    def predict(self, X_test: pd.DataFrame):
        T = len(self.classifiers)
        y_pred = np.zeros(X_test.shape[0])
        for i in range(len(self.classifiers)):
            y_pred += self.classifiers[i].predict(X_test[self.columns[i]])
        return np.sign(y_pred / T)
    
    def _partial_predict(self, df: pd.DataFrame, k:int, y_true):
        y_pred = np.zeros(df.shape[0])
        for i in range(k):
            y_pred += self.classifiers[i].predict(df[self.columns[i]])
        y_pred = np.sign(y_pred / k)
        return self._error_rate(y_true, y_pred)

    def _out_of_bag_obj(self, x_ind: int, X_train):
        score = 0
        cnt = 0

        for i in range(len(self.bacth_indexes)):
            if x_ind not in self.bacth_indexes[i]:

                if isinstance(X_train, dict):
                    X_source = X_train[i]
                else:
                    X_source = X_train

                x_sample = X_source.loc[[x_ind], self.columns[i]]

                score += self.classifiers[i].predict(x_sample)[0]
                cnt += 1

        if cnt != 0:
            return score / cnt
        
        return None

    def out_of_bag_score(self, X_train: pd.DataFrame, y_train: pd.Series):
        res = 0

        for i in X_train.index:
            pred = self._out_of_bag_obj(i, X_train)

            if pred is None:
                continue

            if np.sign(pred) != y_train.loc[i]:
                res += 1

        return res

    def get_params(self, deep=True):
        """Возвращает все параметры модели, включая параметры base_classifier"""
        params = {
            'l_sample': self.l_sample,
            'n_feat': self.n_feat,
            'n_estimators': self.n_estimators,
            'tol_train': self.tol_train,
            'tol_valid': self.tol_valid,
        }

        if deep and hasattr(self.base_classifier, 'get_params'):
            tree_params = self.base_classifier.get_params(deep=deep)
            for key, value in tree_params.items():
                params[f'base_classifier__{key}'] = value

        return params

    def set_params(self, **params):
        """Устанавливает параметры модели, включая параметры base_classifier"""
        tree_params = {}

        for key, value in params.items():
            if key.startswith('base_classifier__'):
                tree_key = key.replace('base_classifier__', '')
                tree_params[tree_key] = value
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")

        if tree_params:
            if hasattr(self.base_classifier, 'set_params'):
                self.base_classifier.set_params(**tree_params)
            else:
                self.base_classifier = DecisionTreeClassifier(**tree_params)

        return self

    def importance(self, col: str, X_train: pd.DataFrame, y_train: pd.Series):
        oob_base = self.out_of_bag_score(X_train, y_train)

        shuffled_datasets = {}

        for i in range(len(self.classifiers)):
            oob_ind = X_train.index.difference(self.bacth_indexes[i])

            X_copy = X_train.copy()

            X_copy.loc[oob_ind, col] = (
                X_copy.loc[oob_ind, col]
                .sample(frac=1, random_state=i)
                .values
            )

            shuffled_datasets[i] = X_copy

        oob_permuted = 0

        for idx in X_train.index:
            pred = self._out_of_bag_obj(idx, shuffled_datasets)

            if pred is None:
                continue

            if np.sign(pred) != y_train.loc[idx]:
                oob_permuted += 1

        if oob_base == 0:
            return float(oob_permuted)
            
        return max(0.0, (oob_permuted - oob_base) / oob_base * 100)

if __name__ == '__main__':
    pass
