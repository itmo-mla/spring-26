import numpy as np
from scipy.sparse import find, csr_matrix


class SVD:
    def __init__(self, n_factors=50, learning_rate=0.005, reg=0.02, n_epochs=30):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        
    def fit(self, R):
        # Конвертация в csr формат
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        
        n_users, n_items = R.shape
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = np.mean(R.data)
        
        rows, cols, ratings = find(R)
        
        for epoch in range(self.n_epochs):
            total_loss = 0
            idx = np.random.permutation(len(ratings))
            
            for idx_i in idx:
                u = rows[idx_i]
                i = cols[idx_i]
                r_ui = ratings[idx_i]
                
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i] + np.dot(self.P[u], self.Q[i])
                err = r_ui - pred
                total_loss += err ** 2
                
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])
                
                p_u = self.P[u].copy()
                q_i = self.Q[i].copy()
                
                self.P[u] += self.lr * (err * q_i - self.reg * p_u)
                self.Q[i] += self.lr * (err * p_u - self.reg * q_i)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.2f}")
    
    def predict(self, R):
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        pred = self.global_mean + self.user_bias[:, None] + self.item_bias[None, :] + self.P @ self.Q.T
        return np.clip(pred, 1, 5)