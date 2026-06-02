import numpy as np
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import joblib

def train_slim(train_matrix, l1_ratio=0.1, alpha=1.0, max_iter=1000, n_jobs=-1):
    n_items = train_matrix.shape[1]
    X = train_matrix.toarray().T

    def fit_item(i):
        y = X[i, :]
        mask = y > 0
        if mask.sum() <= 1:
            return i, np.zeros(n_items)
        y_masked = y[mask]
        X_masked = np.delete(X[:, mask], i, axis=0).T
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter,
                           fit_intercept=False, positive=True)
        model.fit(X_masked, y_masked)
        w = model.coef_
        full_w = np.insert(w, i, 0)
        return i, full_w

    results = joblib.Parallel(n_jobs=n_jobs, backend='loky')(
        joblib.delayed(fit_item)(i) for i in tqdm(range(n_items), desc="SLIM обучение")
    )
    W = np.zeros((n_items, n_items))
    for i, w in results:
        W[:, i] = w
    return W

def als_explicit(R, k=50, reg=0.1, iterations=10, verbose=True):
    n_users, n_items = R.shape
    U = np.random.normal(0, 0.1, (n_users, k))
    V = np.random.normal(0, 0.1, (n_items, k))
    R_dense = R.toarray()
    train_rmse_hist = []

    for it in range(iterations):
        for u in range(n_users):
            items_u = np.where(R_dense[u] > 0)[0]
            if len(items_u) == 0:
                continue
            V_u = V[items_u, :]
            A = V_u.T @ V_u + reg * np.eye(k)
            b = V_u.T @ R_dense[u, items_u]
            U[u] = np.linalg.solve(A, b)

        for i in range(n_items):
            users_i = np.where(R_dense[:, i] > 0)[0]
            if len(users_i) == 0:
                continue
            U_i = U[users_i, :]
            A = U_i.T @ U_i + reg * np.eye(k)
            b = U_i.T @ R_dense[users_i, i]
            V[i] = np.linalg.solve(A, b)

        pred = U @ V.T
        mask = R_dense > 0
        rmse = np.sqrt(np.mean((R_dense[mask] - pred[mask]) ** 2))
        train_rmse_hist.append(rmse)
        if verbose:
            print(f"  ALS итерация {it+1}, train RMSE = {rmse:.4f}")
    return U, V, train_rmse_hist

def predict_and_rmse(W, test_df, train_matrix):
    y_true, y_pred = [], []
    for row in tqdm(test_df.itertuples(), total=len(test_df), desc="Предсказание"):
        u, i, r = int(row.user_id), int(row.item_id), row.rating
        user_vec = train_matrix[u].toarray().flatten()
        pred = user_vec.dot(W[:, i])
        y_true.append(r)
        y_pred.append(pred)
    return np.array(y_true), np.array(y_pred)