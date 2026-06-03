import numpy as np
from collections import Counter

class DecisionNode:
    """Узел дерева решений."""
    def __init__(self, feature=None, threshold=None, values=None, branches=None, is_leaf=False, probabilities=None, default_class=None):
        self.feature = feature              # Индекс признака, по которому идет разбиение
        self.threshold = threshold          # Порог разбиения (для числовых признаков)
        self.values = values                # Список уникальных категорий (для категориальных признаков)
        self.branches = branches or {}      # Словарь поддеревьев
        self.is_leaf = is_leaf              # Флаг: является ли узел листом
        self.probabilities = probabilities  # Вероятности классов в листе
        self.default_class = default_class  # Класс по умолчанию (на случай пустых путей)

class ID3GiniTree:
    """Дерево решений с критерием Джини, поддержкой пропусков и редукцией."""
    def __init__(self, max_depth=None, min_samples_split=2, categorical_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorical_features = categorical_features or []
        self.root = None

    def _gini(self, y, weights):
        """Вычисление критерия Джини с учетом весов объектов."""
        if len(y) == 0:
            return 0
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0
        
        classes = np.unique(y)
        gini_sum = 0
        for cls in classes:
            p = np.sum(weights[y == cls]) / total_weight
            gini_sum += p ** 2
        return 1.0 - gini_sum

    def _split_data(self, X, y, weights, feature, threshold=None):
        """Разбиение данных по признаку с учетом пропущенных значений (NaN)."""
        feature_col = X[:, feature]
        
        nan_mask = np.isnan(feature_col) if not np.issubdtype(feature_col.dtype, np.object_) else (feature_col == None)
        if feature_col.dtype.kind in ['U', 'S', 'O']:
            nan_mask = (feature_col == 'None') | (feature_col == 'nan') | (feature_col == None)
            
        known_mask = ~nan_mask
        
        X_known, y_known, w_known = X[known_mask], y[known_mask], weights[known_mask]
        X_nan, y_nan, w_nan = X[nan_mask], y[nan_mask], weights[nan_mask]
        
        splits = {}
        
        if feature in self.categorical_features:
            unique_vals = np.unique(X_known[:, feature])
            for val in unique_vals:
                val_mask = X_known[:, feature] == val
                splits[val] = (X_known[val_mask], y_known[val_mask], w_known[val_mask])
        else:
            left_mask = X_known[:, feature].astype(float) <= threshold
            right_mask = ~left_mask
            splits['left'] = (X_known[left_mask], y_known[left_mask], w_known[left_mask])
            splits['right'] = (X_known[right_mask], y_known[right_mask], w_known[right_mask])
            
        total_known_weight = np.sum(w_known)
        if total_known_weight > 0 and len(X_nan) > 0:
            for key, (X_s, y_s, w_s) in splits.items():
                branch_weight = np.sum(w_s)
                if branch_weight > 0:
                    prob = branch_weight / total_known_weight
                    X_nan_updated = X_nan.copy()
                    w_nan_updated = w_nan * prob
                    
                    splits[key] = (
                        np.vstack([X_s, X_nan_updated]),
                        np.concatenate([y_s, y_nan]),
                        np.concatenate([w_s, w_nan_updated])
                    )
        return splits

    def _best_split(self, X, y, weights):
        """Поиск лучшего разбиения по критерию Джини."""
        best_gini_gain = -1
        best_criteria = None
        
        current_gini = self._gini(y, weights)
        n_features = X.shape[1]
        
        for feature in range(n_features):
            feature_col = X[:, feature]
            
            nan_mask = (feature_col == 'None') | (feature_col == 'nan') if feature_col.dtype.kind in ['U', 'S', 'O'] else np.isnan(feature_col)
            known_vals = feature_col[~nan_mask]
            
            if len(known_vals) == 0:
                continue
                
            if feature in self.categorical_features:
                splits = self._split_data(X, y, weights, feature)
                weighted_gini = 0
                total_w = np.sum(weights)
                
                for _, (_, y_s, w_s) in splits.items():
                    weighted_gini += (np.sum(w_s) / total_w) * self._gini(y_s, w_s)
                
                gain = current_gini - weighted_gini
                if gain > best_gini_gain:
                    best_gini_gain = gain
                    best_criteria = (feature, None)
            else:
                sorted_vals = np.sort(np.unique(known_vals.astype(float)))
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
                
                for t in thresholds:
                    splits = self._split_data(X, y, weights, feature, threshold=t)
                    weighted_gini = 0
                    total_w = np.sum(weights)
                    
                    for _, (_, y_s, w_s) in splits.items():
                        weighted_gini += (np.sum(w_s) / total_w) * self._gini(y_s, w_s)
                        
                    gain = current_gini - weighted_gini
                    if gain > best_gini_gain:
                        best_gini_gain = gain
                        best_criteria = (feature, t)
                        
        return best_criteria

    def _build_tree(self, X, y, weights, depth=0):
        """Рекурсивный метод построения дерева (ID3)."""
        num_samples = len(y)
        if num_samples == 0:
            return DecisionNode(is_leaf=True, probabilities={})
            
        counts = Counter()
        for cls, w in zip(y, weights):
            counts[cls] += w
        default_class = counts.most_common(1)[0][0]

        total_w = np.sum(weights)
        probs = {cls: w / total_w for cls, w in counts.items()} if total_w > 0 else {}
        
        if (len(np.unique(y)) == 1 or 
            num_samples < self.min_samples_split or 
            (self.max_depth is not None and depth >= self.max_depth)):
            return DecisionNode(is_leaf=True, probabilities=probs, default_class=default_class)
            
        best_criteria = self._best_split(X, y, weights)
        if not best_criteria:
            return DecisionNode(is_leaf=True, probabilities=probs, default_class=default_class)
            
        feature, threshold = best_criteria
        splits = self._split_data(X, y, weights, feature, threshold)
        
        node = DecisionNode(feature=feature, threshold=threshold, default_class=default_class, probabilities=probs)
        if feature in self.categorical_features:
            node.values = list(splits.keys())
            
        for key, (X_s, y_s, w_s) in splits.items():
            if len(y_s) == len(y):
                node.branches[key] = DecisionNode(is_leaf=True, probabilities=probs, default_class=default_class)
            else:
                node.branches[key] = self._build_tree(X_s, y_s, w_s, depth + 1)
                
        return node

    def fit(self, X, y):
        """Обучение дерева решений."""
        weights = np.ones(len(y), dtype=float)
        self.root = self._build_tree(X, y, weights)
        
    def _predict_row(self, node, x):
        """Вероятностное прохождение одного объекта по дереву (для поддержки NaN)."""
        if node.is_leaf:
            return node.probabilities
            
        val = x[node.feature]
        
        is_nan = False
        if str(val) in ['None', 'nan']:
            is_nan = True
        elif isinstance(val, (int, float)) and np.isnan(val):
            is_nan = True
            
        if is_nan:
            aggregated_probs = Counter()
            total_branch_weight = 0
            
            for key, branch in node.branches.items():
                branch_weight = sum(branch.probabilities.values())
                total_branch_weight += branch_weight
                branch_probs = self._predict_row(branch, x)
                
                for cls, prob in branch_probs.items():
                    aggregated_probs[cls] += prob * branch_weight
                    
            if total_branch_weight > 0:
                return {cls: p / total_branch_weight for cls, p in aggregated_probs.items()}
            return node.probabilities

        if node.feature in self.categorical_features:
            if val in node.branches:
                return self._predict_row(node.branches[val], x)
            return node.probabilities
        else:
            if float(val) <= node.threshold:
                return self._predict_row(node.branches['left'], x)
            return self._predict_row(node.branches['right'], x)

    def predict(self, X):
        """Предсказание классов для набора данных."""
        predictions = []
        for x in X:
            probs = self._predict_row(self.root, x)
            if not probs:
                predictions.append(None)
            else:
                predictions.append(max(probs, key=probs.get))
        return np.array(predictions)

    def prune(self, X_val, y_val):
        """Пост-редукция дерева (Post-Pruning) на валидационной выборке."""
        def _prune_node(node):
            for key, branch in list(node.branches.items()):
                if not branch.is_leaf:
                    _prune_node(branch)
            
            if all(branch.is_leaf for branch in node.branches.values()) and not node.is_leaf:
                baseline_preds = self.predict(X_val)
                baseline_acc = np.mean(baseline_preds == y_val)
                
                node.is_leaf = True
                
                pruned_preds = self.predict(X_val)
                pruned_acc = np.mean(pruned_preds == y_val)
                
                if pruned_acc >= baseline_acc:
                    node.branches = {}
                else:
                    node.is_leaf = False
                    
        _prune_node(self.root)

def get_tree_depth(node):
    """Рекурсивный подсчет максимальной глубины дерева."""
    if node is None or node.is_leaf:
        return 0
    if not node.branches:
        return 0
    return 1 + max(get_tree_depth(branch) for branch in node.branches.values())

def get_leaf_count(node):
    """Рекурсивный подсчет общего количества листьев в дереве."""
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return sum(get_leaf_count(branch) for branch in node.branches.values())