import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


class SLIM:
    def __init__(self, alpha=0.01, l1_ratio=0.5, top_k=100, max_iter=2000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.top_k = top_k
        self.max_iter = max_iter
        self.W = None
        
    def fit(self, R):
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        
        n_items = R.shape[1]
        # Используем lil_matrix для эффективного изменения структуры
        self.W = lil_matrix((n_items, n_items))
        
        # Нормализация по столбцам для сходства товаров
        R_norm_items = normalize(R, norm='l2', axis=0)
        item_sim = R_norm_items.T @ R_norm_items
        
        for j in range(n_items):
            sim_j = item_sim[j].toarray().ravel()
            top_indices = np.argsort(sim_j)[-(self.top_k+1):-1]
            top_indices = top_indices[top_indices != j]
            
            if len(top_indices) == 0:
                continue
            
            X = R[:, top_indices].toarray()
            y = R[:, j].toarray().ravel()
            
            mask = y != 0
            if mask.sum() < 10:
                continue
            
            X = X[mask]
            y = y[mask]
            
            # Увеличиваем max_iter и подбираем параметры
            enet = ElasticNet(
                alpha=self.alpha, 
                l1_ratio=self.l1_ratio, 
                fit_intercept=False, 
                max_iter=self.max_iter,
                tol=1e-3  # Увеличиваем допуск для более быстрой сходимости
            )
            enet.fit(X, y)
            
            w_j = enet.coef_
            w_j = np.maximum(w_j, 0)
            
            # Записываем в lil_matrix (эффективнее для построения)
            for idx, val in zip(top_indices, w_j):
                if val > 1e-6:
                    self.W[idx, j] = val
            
            if j % 100 == 0 and j > 0:
                print(f"Обработано товаров: {j}/{n_items}")
        
        # Конвертируем в csr и нормализуем столбцы
        self.W = self.W.tocsr()
        col_sums = self.W.sum(axis=0)
        col_sums = np.array(col_sums).ravel()
        col_sums[col_sums == 0] = 1
        self.W = self.W @ diags(1.0 / col_sums)
        
        print("SLIM обучен")
    
    def predict(self, R):
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        result = R @ self.W
        return result
    