from collections import Counter
import math


class Leaf:
    """Лист дерева"""
    
    def __init__(self, value):
        self.value = value
    
    def predict(self, x):
        return self.value
    
    def get_distribution(self):
        """Возвращает распределение классов"""
        return {self.value: 1.0}


class Node:
    """Узел дерева"""
    
    def __init__(self, feature, threshold, left, right, left_weight, right_weight):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.left_weight = left_weight
        self.right_weight = right_weight
    
    def predict(self, x):
        """
        Предсказание с обработкой пропусков через взвешенное распределение
        """
        feature_value = x[self.feature]

        if feature_value is None:
            left_dist = self._get_weighted_distribution(self.left)
            right_dist = self._get_weighted_distribution(self.right)

            combined_dist = Counter()
            for cls, prob in left_dist.items():
                combined_dist[cls] += self.left_weight * prob
            for cls, prob in right_dist.items():
                combined_dist[cls] += self.right_weight * prob
            
            return combined_dist.most_common(1)[0][0]
        
        if feature_value <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)
    
    def _get_weighted_distribution(self, node):
        """Рекурсивный сбор взвешенного распределения"""
        if isinstance(node, Leaf):
            return node.get_distribution()
        
        left_dist = node._get_weighted_distribution(node.left)
        right_dist = node._get_weighted_distribution(node.right)
        
        combined_dist = Counter()
        for cls, prob in left_dist.items():
            combined_dist[cls] += node.left_weight * prob
        for cls, prob in right_dist.items():
            combined_dist[cls] += node.right_weight * prob
        
        return combined_dist


class DecisionTree:
    """Decision Tree Classifier"""
    
    def __init__(self, max_depth=5, min_samples_split=2, criterion='gini', min_gain=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.min_gain = min_gain
        self.tree = None
        self.n_classes = None
    
    def fit(self, X, y):
        """Построение дерева"""
        self.n_classes = len(set(y))
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        """Рекурсивная функция построения дерева"""
        n_samples = len(y)
        n_classes = len(set(y))
        
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            return Leaf(self._most_common_class(y))
        
        best_feature, best_threshold, best_gain = None, None, self.min_gain
        
        for feature in range(len(X[0])):
            thresholds = sorted(set(x[feature] for x in X if x[feature] is not None))
            
            for threshold in thresholds:
                left_idx = [i for i, x in enumerate(X) if x[feature] is not None and x[feature] <= threshold]
                right_idx = [i for i, x in enumerate(X) if x[feature] is not None and x[feature] > threshold]
                
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                
                y_left = [y[i] for i in left_idx]
                y_right = [y[i] for i in right_idx]
                
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_feature is None:
            return Leaf(self._most_common_class(y))
        
        left_X = [X[i] for i in range(len(X)) if X[i][best_feature] is not None and X[i][best_feature] <= best_threshold]
        left_y = [y[i] for i in range(len(y)) if X[i][best_feature] is not None and X[i][best_feature] <= best_threshold]
        right_X = [X[i] for i in range(len(X)) if X[i][best_feature] is not None and X[i][best_feature] > best_threshold]
        right_y = [y[i] for i in range(len(y)) if X[i][best_feature] is not None and X[i][best_feature] > best_threshold]
        
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right
        left_weight = n_left / n_total if n_total > 0 else 0.5
        right_weight = n_right / n_total if n_total > 0 else 0.5
        
        return Node(best_feature, best_threshold, left_child, right_child, left_weight, right_weight)
    
    def predict(self, X):
        """Предсказание классов"""
        return [self._predict_sample(x) for x in X]
    
    def _predict_sample(self, x):
        """Предсказание для одного объекта"""
        return self.tree.predict(x)
    
    def _most_common_class(self, y):
        """Возвращает наиболее частый класс"""
        return Counter(y).most_common(1)[0][0]
    
    def _gini(self, y):
        """Вычисление критерия Джини"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        impurity = 1.0
        
        for count in counts.values():
            prob = count / len(y)
            impurity -= prob ** 2
        
        return impurity
    
    def _entropy(self, y):
        """Вычисление энтропии"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        impurity = 0.0
        
        for count in counts.values():
            prob = count / len(y)
            impurity -= prob * math.log2(prob)
        
        return impurity
    
    def _impurity(self, y):
        """Вычисление критерия"""
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, y, y_left, y_right):
        """Вычисление прироста информации"""
        n_left = len(y_left)
        n_right = len(y_right)
        n_valid = n_left + n_right
        
        if n_valid == 0:
            return 0
        
        parent_impurity = self._impurity(y)
        weighted_child_impurity = (n_left / n_valid) * self._impurity(y_left) + (n_right / n_valid) * self._impurity(y_right)
        
        return parent_impurity - weighted_child_impurity
    
    # ==================== POST-PRUNING (REP) ====================
    
    def prune(self, X_val, y_val):
        """Reduced Error Pruning (REP)"""
        self.tree = self._prune_node(self.tree, X_val, y_val)
        return self
    
    def _prune_node(self, node, X_val, y_val):
        """Рекурсивное усечение узла"""
        if isinstance(node, Leaf):
            return node
        
        left_X_val = [x for x, y in zip(X_val, y_val) 
                      if x[node.feature] is not None and x[node.feature] <= node.threshold]
        left_y_val = [y for x, y in zip(X_val, y_val) 
                      if x[node.feature] is not None and x[node.feature] <= node.threshold]
        
        right_X_val = [x for x, y in zip(X_val, y_val) 
                       if x[node.feature] is not None and x[node.feature] > node.threshold]
        right_y_val = [y for x, y in zip(X_val, y_val) 
                       if x[node.feature] is not None and x[node.feature] > node.threshold]
        
        if not isinstance(node.left, Leaf):
            node.left = self._prune_node(node.left, left_X_val, left_y_val)
        
        if not isinstance(node.right, Leaf):
            node.right = self._prune_node(node.right, right_X_val, right_y_val)
        
        error_before = self._calculate_error(X_val, y_val)
        
        majority_class = self._most_common_class(y_val) if len(y_val) > 0 else 0
        error_after = sum(1 for true in y_val if true != majority_class) / len(y_val) if len(y_val) > 0 else 0
        
        if error_after <= error_before:
            return Leaf(majority_class)
        
        return node
    
    def _calculate_error(self, X, y):
        """Вычисление доли ошибок"""
        if len(y) == 0:
            return 0
        
        predictions = self.predict(X)
        errors = sum(1 for pred, true in zip(predictions, y) if pred != true)
        return errors / len(y)
