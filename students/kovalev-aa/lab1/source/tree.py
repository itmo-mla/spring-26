import numpy as np
import pandas as pd

class Leaf:
    def __init__(self, value):
        self.value = value

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class Tree:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def train(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None, ccp_alpha=0.0):
        self.root = self.__build_tree(X, y, depth=0)
        if X_val is not None and y_val is not None and ccp_alpha > 0:
            self.root = self.__prune(self.root, X_val, y_val, ccp_alpha)

    def __gini(self, y):
        if len(y) == 0:
            return 0
        classes = np.unique(y)
        gini = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            gini -= p ** 2
        return gini

    def __best_split(self, X, y):
        best_gini = float('inf')
        best_feat = None
        best_thresh = None

        for feat in X.columns:
            values = np.unique(X[feat])
            for thresh in values:  # проверяем каждый уникальный порог
                left_y = y[X[feat] <= thresh]
                right_y = y[X[feat] > thresh]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gini_split = (len(left_y)/len(y))*self.__gini(left_y) + (len(right_y)/len(y))*self.__gini(right_y)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh

    def __build_tree(self, X, y, depth=0):
        # Если все метки одинаковые - лист
        if len(np.unique(y)) == 1:
            return Leaf(y.iloc[0])

        # Если мало примеров или достигли максимальной глубины - лист с большинством
        if len(X) <= 1 or (self.max_depth is not None and depth >= self.max_depth):
            return Leaf(y.value_counts().idxmax())

        best_feat, best_thresh = self.__best_split(X, y)
        if best_feat is None:
            return Leaf(y.value_counts().idxmax())

        left_mask = X[best_feat] <= best_thresh
        right_mask = X[best_feat] > best_thresh

        left = self.__build_tree(X[left_mask], y[left_mask], depth+1)
        right = self.__build_tree(X[right_mask], y[right_mask], depth+1)

        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def __predict_one(self, x, node):
        if isinstance(node, Leaf):
            return node.value
        if x[node.feature] <= node.threshold:
            return self.__predict_one(x, node.left)
        else:
            return self.__predict_one(x, node.right)

    def predict(self, X: pd.DataFrame):
        return X.apply(lambda row: self.__predict_one(row, self.root), axis=1)

    def __prune(self, node, X_val, y_val, ccp_alpha):
        if isinstance(node, Leaf):
            return node

        left_mask = X_val[node.feature] <= node.threshold
        right_mask = X_val[node.feature] > node.threshold

        node.left = self.__prune(node.left, X_val[left_mask], y_val[left_mask], ccp_alpha)
        node.right = self.__prune(node.right, X_val[right_mask], y_val[right_mask], ccp_alpha)

        if len(y_val) == 0:
            return node

        y_pred = X_val.apply(lambda row: self.__predict_one(row, node), axis=1)
        error_subtree = np.mean(y_pred != y_val)

        majority_class = y_val.value_counts().idxmax()
        error_leaf = np.mean([majority_class]*len(y_val) != y_val)

        if error_leaf + ccp_alpha <= error_subtree:
            return Leaf(majority_class)

        return node