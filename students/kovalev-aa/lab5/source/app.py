"""
SLIM and Latent Semantic Model for Collaborative Filtering
Comparison with Surprise and Reference SLIM Implementation
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
import time
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from surprise import Dataset, Reader, SVD as SurpriseSVD
from surprise.model_selection import train_test_split as surprise_split
from surprise import accuracy


class SLIM:
    """Our SLIM implementation with Elastic Net regularization"""

    def __init__(self, alpha=0.001, l1_ratio=0.5, max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.W = None

    def _fit_item_model(self, X, item_idx):
        n_items = X.shape[1]
        y = X[:, item_idx].toarray().ravel()
        cols = [c for c in range(n_items) if c != item_idx]
        X_features = X[:, cols]

        mask = y != 0
        if np.sum(mask) < 2:
            return np.zeros(n_items)

        X_features = X_features[mask]
        y = y[mask]

        if X_features.shape[1] == 0:
            return np.zeros(n_items)

        X_dense = X_features.toarray()
        w = np.zeros(X_features.shape[1])
        XtX = X_dense.T @ X_dense
        Xty = X_dense.T @ y
        n_samples = len(y)

        for _ in range(self.max_iter):
            w_old = w.copy()
            for j in range(len(w)):
                if XtX[j, j] == 0:
                    continue
                rho = Xty[j] - (XtX[j] @ w - XtX[j, j] * w[j])

                if self.l1_ratio > 0:
                    l1 = self.alpha * self.l1_ratio * n_samples
                    l2 = self.alpha * (1 - self.l1_ratio) * n_samples
                    if rho > l1:
                        w[j] = (rho - l1) / (XtX[j, j] + l2)
                    elif rho < -l1:
                        w[j] = (rho + l1) / (XtX[j, j] + l2)
                    else:
                        w[j] = 0
                else:
                    w[j] = rho / (XtX[j, j] + self.alpha * n_samples)

            if np.linalg.norm(w - w_old, 1) < self.tol:
                break

        result = np.zeros(n_items)
        result[cols] = w
        return result

    def fit(self, X):
        n_users, n_items = X.shape
        print(f"Fitting SLIM on {n_users} users, {n_items} items")
        self.W = np.zeros((n_items, n_items))

        for i in range(n_items):
            if i % 20 == 0:
                print(f"  Item {i}/{n_items}")
            self.W[i] = self._fit_item_model(X, i)
        return self

    def predict(self, X_train, user_idx, item_idx):
        user_ratings = X_train[user_idx].toarray().ravel()
        sim = self.W[item_idx]
        mask = user_ratings != 0

        if not mask.any():
            return 3.0

        pred = np.sum(user_ratings[mask] * sim[mask])
        sim_sum = np.sum(np.abs(sim[mask]))
        if sim_sum > 0:
            pred /= sim_sum
        return np.clip(pred, 1, 5)

    def predict_all(self, X_train, test_pairs):
        return [self.predict(X_train, u, i) for u, i in test_pairs]


class ReferenceSLIM:
    """Reference SLIM implementation using sklearn's ElasticNet"""

    def __init__(self, alpha=0.1, l1_ratio=0.1, max_iter=100):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.W = None

    def fit(self, X):
        X = X.astype(np.float32)
        n_items = X.shape[1]
        self.W = np.zeros((n_items, n_items))

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        for j in range(n_items):
            y = X[:, j].toarray().ravel()
            X_j = X.copy()
            X_j[:, j] = 0

            model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                              fit_intercept=False, max_iter=self.max_iter)
            model.fit(X_j, y)
            self.W[:, j] = model.coef_

            if (j + 1) % 20 == 0:
                print(f"  Reference SLIM: item {j+1}/{n_items}")

        return self

    def predict_all(self, X_train, test_pairs):
        predictions = []
        for user, item in test_pairs:
            user_ratings = X_train[user].toarray().ravel()
            pred = np.dot(user_ratings, self.W[:, item])
            sim_sum = np.sum(np.abs(self.W[:, item]))
            if sim_sum > 0:
                pred /= sim_sum
            predictions.append(np.clip(pred, 1, 5))
        return predictions


class LatentSemanticModel:
    """SVD-based latent factor model"""

    def __init__(self, n_factors=20, n_iter=50, lr=0.005, reg_user=0.02, reg_item=0.02):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.lr = lr
        self.reg_user = reg_user
        self.reg_item = reg_item

    def fit(self, X, verbose=False):
        n_users, n_items = X.shape
        ratings = [(u, i, X[u, i]) for u in range(n_users) for i in range(n_items) if X[u, i] != 0]

        self.global_mean = np.mean([r for _, _, r in ratings])
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        for epoch in range(self.n_iter):
            np.random.shuffle(ratings)
            error_sum = 0

            for u, i, r in ratings:
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i] + \
                       np.dot(self.user_factors[u], self.item_factors[i])
                error = r - pred
                error_sum += error ** 2

                self.user_bias[u] += self.lr * (error - self.reg_user * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg_item * self.item_bias[i])

                u_factors = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg_user * u_factors)
                self.item_factors[i] += self.lr * (error * u_factors - self.reg_item * self.item_factors[i])

            if verbose and (epoch + 1) % 20 == 0:
                rmse = np.sqrt(error_sum / len(ratings))
                print(f"  Epoch {epoch+1}/{self.n_iter}, RMSE: {rmse:.4f}")
        return self

    def predict(self, user_idx, item_idx):
        pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx] + \
               np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return np.clip(pred, 1, 5)

    def predict_all(self, test_pairs):
        return [self.predict(u, i) for u, i in test_pairs]


def ndcg_at_k(true_ratings, pred_ratings, k=10):
    """Calculate NDCG@k for recommendations"""
    nDCG = 0.0
    for i in range(len(true_ratings)):
        # Get top k predictions
        top_k_idx = np.argsort(pred_ratings[i])[-k:][::-1]

        # Get true ratings for top k items
        true_k = true_ratings[i][top_k_idx]

        # DCG
        dcg = 0.0
        for j, rel in enumerate(true_k):
            dcg += (2**rel - 1) / np.log2(j + 2)

        # IDCG (ideal DCG)
        ideal_true = np.sort(true_ratings[i])[-k:][::-1]
        idcg = 0.0
        for j, rel in enumerate(ideal_true):
            idcg += (2**rel - 1) / np.log2(j + 2)

        if idcg > 0:
            nDCG += dcg / idcg

    return nDCG / len(true_ratings)


def evaluate_model(model, X_train, test_df, n_users, n_items, model_name="Model"):
    """Evaluate model on test data"""
    test_pairs = list(zip(test_df['user'].values, test_df['item'].values))
    true_ratings = test_df['rating'].values

    # Get predictions
    if hasattr(model, 'predict_all'):
        try:
            preds = model.predict_all(X_train, test_pairs)
        except TypeError:
            preds = model.predict_all(test_pairs)
    else:
        preds = model.predict_all(test_pairs)

    # RMSE and MAE
    rmse = np.sqrt(mean_squared_error(true_ratings, preds))
    mae = np.mean(np.abs(true_ratings - preds))

    # NDCG@10 - Build user-item matrix for test users
    test_users = test_df['user'].unique()

    # Get max indices
    max_user = int(test_users.max()) + 1
    max_item = n_items

    user_true = {}
    user_pred = {}

    for user in test_users:
        user_true[user] = np.zeros(max_item)
        user_pred[user] = np.zeros(max_item)

        # Get true ratings for this user
        user_test = test_df[test_df['user'] == user]
        for _, row in user_test.iterrows():
            item_idx = int(row['item'])
            user_true[user][item_idx] = row['rating']

        # Get predictions for all items for this user
        for item in range(max_item):
            try:
                if hasattr(model, 'predict'):
                    try:
                        user_pred[user][item] = model.predict(X_train, user, item)
                    except TypeError:
                        user_pred[user][item] = model.predict(user, item)
                else:
                    user_pred[user][item] = model.predict(user, item)
            except:
                user_pred[user][item] = 3.0

    # Calculate NDCG for each user
    true_matrix = np.array([user_true[u] for u in test_users])
    pred_matrix = np.array([user_pred[u] for u in test_users])
    ndcg_score = ndcg_at_k(true_matrix, pred_matrix, k=10)

    return rmse, mae, ndcg_score


def load_dataset(path=None, max_users=500, max_items=300, min_ratings=5):
    """Load and trim MovieLens dataset"""
    if path is None:
        possible = ['ml-latest-small/ratings.csv', 'ratings.csv', 'u.data']
        for p in possible:
            if os.path.exists(p):
                path = p
                break

    if path and os.path.exists(path):
        if path.endswith('.u.data'):
            df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
        else:
            df = pd.read_csv(path)
            if 'userId' in df.columns:
                df = df.rename(columns={'userId': 'user', 'movieId': 'item'})
    else:
        np.random.seed(42)
        data = []
        for u in range(max_users):
            for _ in range(np.random.randint(min_ratings, min_ratings + 10)):
                i = np.random.randint(max_items)
                r = np.random.choice([1,2,3,4,5], p=[0.05,0.1,0.2,0.35,0.3])
                data.append([u, i, r])
        df = pd.DataFrame(data, columns=['user', 'item', 'rating'])

    # Filter and remap
    user_counts = df.groupby('user').size()
    item_counts = df.groupby('item').size()
    df = df[df['user'].isin(user_counts[user_counts >= min_ratings].index)]
    df = df[df['item'].isin(item_counts[item_counts >= 3].index)]

    top_users = df.groupby('user').size().nlargest(max_users).index
    top_items = df.groupby('item').size().nlargest(max_items).index
    df = df[df['user'].isin(top_users) & df['item'].isin(top_items)]

    # Remap to 0-index integers
    unique_users = df['user'].unique()
    unique_items = df['item'].unique()
    user_map = {old: new for new, old in enumerate(unique_users)}
    item_map = {old: new for new, old in enumerate(unique_items)}

    df['user'] = df['user'].map(user_map)
    df['item'] = df['item'].map(item_map)

    print(f"Dataset: {len(df)} ratings, {df['user'].nunique()} users, {df['item'].nunique()} items")
    sparsity = (1 - len(df)/(df['user'].nunique()*df['item'].nunique()))*100
    print(f"Sparsity: {sparsity:.1f}%")
    return df


def main():
    print("=" * 80)
    print("SLIM & Latent Semantic Model - Complete Comparison")
    print("=" * 80)

    # Load data
    print("\n1. Loading dataset...")
    df = load_dataset(max_users=1000, max_items=500, min_ratings=5)  # Уменьшил для скорости

    # Train/test split
    print("\n2. Train/test split...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"   Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")

    # Prepare matrices
    n_users = df['user'].nunique()
    n_items = df['item'].nunique()
    X_train = csr_matrix((train_df['rating'].values,
                          (train_df['user'].values, train_df['item'].values)),
                         shape=(n_users, n_items))

    # Train and evaluate all models
    print("\n3. Training models...")

    # Our SLIM
    print("\n  Training Our SLIM...")
    slim = SLIM(alpha=0.001, l1_ratio=0.5, max_iter=30)
    slim_start = time.time()
    slim.fit(X_train)
    slim_time = time.time() - slim_start
    slim_rmse, slim_mae, slim_ndcg = evaluate_model(slim, X_train, test_df, n_users, n_items, "Our SLIM")

    # Reference SLIM
    print("\n  Training Reference SLIM...")
    ref_slim = ReferenceSLIM(alpha=0.1, l1_ratio=0.1, max_iter=100)
    ref_start = time.time()
    ref_slim.fit(X_train)
    ref_time = time.time() - ref_start
    ref_rmse, ref_mae, ref_ndcg = evaluate_model(ref_slim, X_train, test_df, n_users, n_items, "Reference SLIM")

    # Our LSM
    print("\n  Training Our LSM...")
    lsm = LatentSemanticModel(n_factors=20, n_iter=50, lr=0.005)
    lsm_start = time.time()
    lsm.fit(X_train, verbose=False)
    lsm_time = time.time() - lsm_start
    lsm_rmse, lsm_mae, lsm_ndcg = evaluate_model(lsm, X_train, test_df, n_users, n_items, "Our LSM")

    svd_rmse = svd_mae = svd_ndcg = svd_time = None
    print("\n  Training Surprise SVD...")
    reader = Reader(rating_scale=(1, 5))
    all_ratings = pd.concat([train_df, test_df])
    data = Dataset.load_from_df(all_ratings[['user', 'item', 'rating']], reader)
    trainset, testset = surprise_split(data, test_size=len(test_df), random_state=42)

    svd = SurpriseSVD(n_factors=20, n_epochs=50, lr_all=0.005, reg_all=0.02)
    svd_start = time.time()
    svd.fit(trainset)
    svd_time = time.time() - svd_start

    # Convert surprise predictions to arrays for NDCG
    svd_preds = svd.test(testset)
    svd_pred_dict = {}
    for pred in svd_preds:
        svd_pred_dict[(pred.uid, pred.iid)] = pred.est

    # Calculate RMSE and MAE
    svd_rmse = accuracy.rmse(svd_preds, verbose=False)
    svd_mae = accuracy.mae(svd_preds, verbose=False)

    # Calculate NDCG for Surprise
    test_users = test_df['user'].unique()
    user_true = {}
    user_pred = {}

    for user in test_users:
        user_true[user] = np.zeros(n_items)
        user_pred[user] = np.zeros(n_items)

        user_test = test_df[test_df['user'] == user]
        for _, row in user_test.iterrows():
            user_true[user][int(row['item'])] = row['rating']

        for item in range(n_items):
            user_pred[user][item] = svd_pred_dict.get((user, item), 3.0)

    true_matrix = np.array([user_true[u] for u in test_users])
    pred_matrix = np.array([user_pred[u] for u in test_users])
    svd_ndcg = ndcg_at_k(true_matrix, pred_matrix, k=10)

    # FINAL COMPARISON TABLE
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"\n{'Model':<35} {'RMSE':>8} {'MAE':>8} {'NDCG@10':>10} {'Time(s)':>8}")
    print("-" * 70)
    print(f"{'Our SLIM':<35} {slim_rmse:>8.4f} {slim_mae:>8.4f} {slim_ndcg:>10.4f} {slim_time:>8.2f}")
    print(f"{'Reference SLIM':<35} {ref_rmse:>8.4f} {ref_mae:>8.4f} {ref_ndcg:>10.4f} {ref_time:>8.2f}")
    print(f"{'Our LSM (SVD)':<35} {lsm_rmse:>8.4f} {lsm_mae:>8.4f} {lsm_ndcg:>10.4f} {lsm_time:>8.2f}")
    print(f"{'Surprise SVD':<35} {svd_rmse:>8.4f} {svd_mae:>8.4f} {svd_ndcg:>10.4f} {svd_time:>8.2f}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()