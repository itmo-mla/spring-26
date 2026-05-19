import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Patch
import numpy.ctypeslib as npct
import time
import os
import urllib.request
import zipfile
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'output'
os.makedirs(OUT_DIR, exist_ok=True)
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

SLIM_LIB_PATH = 'libslim/libslim.so'


def download_movielens(path='ml-100k'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = os.path.join(OUT_DIR, 'ml-100k.zip')
    data_path = os.path.join(OUT_DIR, path)

    if not os.path.exists(os.path.join(data_path, 'u.data')):
        print('Загрузка MovieLens-100K...')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(OUT_DIR)
        print('Загрузка завершена')
    else:
        print('Датасет уже загружен')
    return data_path


def load_movielens(data_path):
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(os.path.join(data_path, 'u.data'), sep='\t', names=cols)

    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1

    return df


def build_sparse_matrix(df, n_users, n_items):
    R = csr_matrix((df['rating'].values, (df['user_id'].values, df['item_id'].values)),
                   shape=(n_users, n_items)
                   )
    return R


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance)[:k]
    if relevance.size:
        return np.sum((2 ** relevance - 1) / np.log2(np.arange(2, relevance.size + 2)))
    return 0.0


def ndcg_at_k(relevance, k):
    dcg = dcg_at_k(relevance, k)
    ideal = dcg_at_k(np.sort(relevance)[::-1], k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def compute_ndcg(model, R_train, R_test, k=10, sample_users=None):
    """
    NDCG@k для каждого пользователя

    Для тестовых рейтингов считаем релевантность = рейтинг - 3 + 1
    (шкала 1-5 => релевантность 0-3, порог 3).
    """
    n_users = R_train.shape[0]
    if sample_users is None:
        sample_users = np.arange(n_users)

    ndcg_scores = []
    for u in sample_users:
        # сформированные рекомендации (top-k)
        try:
            scores = model.predict_user(u, R=R_train)
        except TypeError:
            scores = model.predict_user(u)

        train_items = R_train[u].nonzero()[1]
        scores[train_items] = -np.inf
        top_k = np.argsort(scores)[::-1][:k]

        # релевантность топ-k в тесте
        relevance = np.zeros(k)
        test_items = R_test[u].nonzero()[1]
        test_ratings = R_test[u, test_items].toarray().flatten()

        # релевантность: binarize по порогу 3
        for idx, item in enumerate(top_k):
            if item in test_items:
                rel = test_ratings[np.where(test_items == item)[0][0]]
                relevance[idx] = rel  # используем сырой рейтинг как релевантность

        if relevance.max() > 0:
            ndcg_scores.append(ndcg_at_k(relevance, k))

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


class MySLIM:
    def __init__(self, alpha=0.01, l1_ratio=0.5, n_neighbors=50):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_neighbors = n_neighbors
        self.W = None
        self.user_means = None
        self.global_mean = None

    def fit(self, R):
        n_users, n_items = R.shape
        self.W = np.zeros((n_items, n_items))
        R_csr = R.tocsr()
        R_csc = R.tocsc()

        # mean centering
        self.global_mean = R.data.mean() if R.nnz > 0 else 0
        self.user_means = np.zeros(n_users)
        for u in range(n_users):
            items = R_csr[u].indices
            if items.size > 0:
                self.user_means[u] = R_csr[u].data.astype(float).mean()
            else:
                self.user_means[u] = self.global_mean

        # центрируем матрицу R_centered = R - user_means
        R_csr_float = R_csr.astype(float)
        ptr = 0
        for u in range(n_users):
            nnz_u = R_csr_float[u].nnz
            if nnz_u > 0:
                R_csr_float.data[ptr:ptr + nnz_u] -= self.user_means[u]
                ptr += nnz_u
        R_csr = R_csr_float
        R_csc = R_csr.tocsc()

        # предвычисляем косинусное сходство для отбора соседей
        print('[SLIM] Вычисление косинусного сходства...')
        norms = np.sqrt(R_csr.multiply(R_csr).sum(axis=0)).A1
        norms[norms == 0] = 1
        R_normed = R_csr.copy()
        R_normed = R_normed.multiply(1.0 / norms)
        sim_matrix = (R_normed.T @ R_normed).toarray()
        np.fill_diagonal(sim_matrix, 0)

        print(f'[SLIM] Обучение {n_items} ElasticNet-регрессий (alpha={self.alpha}, l1_ratio={self.l1_ratio})...')
        start = time.time()
        for i in range(n_items):
            if i % 100 == 0:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (n_items - i - 1)
                print(f'  item {i}/{n_items} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s')

            # целевой вектор: столбец i (рейтинги item i от всех пользователей)
            r_i = np.array(R_csc[:, i].todense(), dtype=float).flatten()

            # отбираем k ближайших соседей по косинусному сходству
            neighbors = np.argsort(sim_matrix[i])[::-1][:self.n_neighbors]

            # фичи: столбцы R для соседних items (без i)
            X = R_csr[:, neighbors].astype(float).toarray()

            # ElasticNet (L1 + L2) регрессия
            reg = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=False,
                             positive=True, max_iter=500, random_state=42)
            reg.fit(X, r_i)
            self.W[i, neighbors] = reg.coef_

        elapsed = time.time() - start
        print(f'[SLIM] Обучение завершено за {elapsed:.1f}s')
        return self

    def predict(self, R):
        """Предсказание. R_hat = (R - user_means) @ W^T + user_means, clipped [1, 5]"""
        R_csr = R.tocsr().astype(float)
        ptr = 0
        for u in range(R_csr.shape[0]):
            nnz_u = R_csr[u].nnz
            if nnz_u > 0:
                R_csr.data[ptr:ptr + nnz_u] -= self.user_means[u]
                ptr += nnz_u
        preds = R_csr @ self.W.T + self.user_means.reshape(-1, 1)
        return np.clip(preds, 1.0, 5.0)

    def predict_user(self, user_idx, R=None):
        """Предсказания для одного пользователя. (r_u - mu_u) @ W^T + mu_u, clipped [1, 5]"""
        if R is None:
            raise ValueError("Передайте матрицу R")
        r_u = np.array(R[user_idx].todense(), dtype=float).flatten()
        r_u_centered = r_u.copy()
        train_items = R[user_idx].nonzero()[1]
        r_u_centered[train_items] -= self.user_means[user_idx]
        preds = r_u_centered @ self.W.T + self.user_means[user_idx]
        return np.clip(preds, 1.0, 5.0)


class MyALS:
    def __init__(self, n_factors=20, lambda_reg=0.1, mu_reg=0.1, n_iter=20):
        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg
        self.n_iter = n_iter
        self.P = None
        self.Q = None
        self.global_mean = None
        self.train_rmse_history = []

    def fit(self, R):
        n_users, n_items = R.shape
        np.random.seed(42)

        self.global_mean = R.data.mean() if R.nnz > 0 else 0

        self.P = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.01, (n_items, self.n_factors))

        R_csr = R.tocsr()
        R_csc = R.tocsc()

        print(f'[ALS] Обучение: {self.n_factors} факторов, {self.n_iter} итераций')
        start = time.time()

        for it in range(self.n_iter):
            # Шаг P (fix Q)
            QtQ = self.Q.T @ self.Q + self.lambda_reg * np.eye(self.n_factors)
            for u in range(n_users):
                items_u = R_csr[u].indices
                if items_u.size == 0:
                    continue
                r_u = R_csr[u].data.astype(float)
                # Вычитаем глобальное среднее
                r_u_centered = r_u - self.global_mean
                Q_u = self.Q[items_u]
                self.P[u] = np.linalg.solve(QtQ, Q_u.T @ r_u_centered)

            # Шаг Q (fix P)
            PtP = self.P.T @ self.P + self.mu_reg * np.eye(self.n_factors)
            for i in range(n_items):
                users_i = R_csc[:, i].indices
                if users_i.size == 0:
                    continue
                r_i = R_csc[:, i].data.astype(float)
                r_i_centered = r_i - self.global_mean
                P_i = self.P[users_i]
                self.Q[i] = np.linalg.solve(PtP, P_i.T @ r_i_centered)

            elapsed = time.time() - start
            if (it + 1) % 5 == 0 or it == 0:
                train_rmse = self._compute_rmse(R_csr)
                self.train_rmse_history.append(train_rmse)
                print(f'  iter {it+1:3d}/{self.n_iter} | train RMSE = {train_rmse:.4f} | time {elapsed:.1f}s')

        elapsed = time.time() - start
        print(f'[ALS] Обучение завершено за {elapsed:.1f}s')
        return self

    def _compute_rmse(self, R_csr):
        errors = []
        for u in range(R_csr.shape[0]):
            items = R_csr[u].indices
            if items.size == 0:
                continue
            r_true = R_csr[u].data.astype(float)
            r_pred = self.P[u] @ self.Q[items].T + self.global_mean
            errors.extend((r_true - r_pred) ** 2)
        return np.sqrt(np.mean(errors)) if errors else 0.0

    def predict(self, R):
        """Полная матрица предсказаний"""
        return self.P @ self.Q.T + self.global_mean

    def predict_user(self, user_idx, R=None):
        """Предсказания для одного пользователя"""
        return self.P[user_idx] @ self.Q.T + self.global_mean


class ReferenceSVD:
    def __init__(self, n_factors=20):
        self.n_factors = n_factors
        self.P = None
        self.Q = None
        self.sigma = None
        self.global_mean = None

    def fit(self, R):
        self.global_mean = R.data.mean() if R.nnz > 0 else 0
        R_centered = R.copy().astype(float)
        R_centered.data -= self.global_mean

        U, sigma, Vt = svds(R_centered, k=self.n_factors)
        # сортируем по убыванию сингулярных чисел
        order = np.argsort(sigma)[::-1]
        U = U[:, order]
        sigma = sigma[order]
        Vt = Vt[order, :]

        self.sigma = sigma
        self.P = U * np.sqrt(sigma)   # (n_users, k)
        self.Q = (Vt.T * np.sqrt(sigma))  # (n_items, k)
        print(f'[Ref-SVD] Обучено {self.n_factors} факторов')
        return self

    def predict(self, R):
        return self.P @ self.Q.T + self.global_mean

    def predict_user(self, user_idx, R=None):
        return self.P[user_idx] @ self.Q.T + self.global_mean


class ReferenceSLIM_Scipy:
    def __init__(self, alpha=0.01, l1_reg=0.01, l2_reg=0.01, n_neighbors=50):
        self.alpha = alpha
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.n_neighbors = n_neighbors
        self.W = None
        self.user_means = None
        self.global_mean = None

    def fit(self, R):
        from scipy.optimize import minimize
        n_users, n_items = R.shape
        self.W = np.zeros((n_items, n_items))
        R_csr = R.tocsr()
        R_csc = R.tocsc()

        self.global_mean = R.data.mean() if R.nnz > 0 else 0
        self.user_means = np.zeros(n_users)
        for u in range(n_users):
            items = R_csr[u].indices
            if items.size > 0:
                self.user_means[u] = R_csr[u].data.astype(float).mean()
            else:
                self.user_means[u] = self.global_mean

        R_csr_float = R_csr.astype(float)
        ptr = 0
        for u in range(n_users):
            nnz_u = R_csr_float[u].nnz
            if nnz_u > 0:
                R_csr_float.data[ptr:ptr + nnz_u] -= self.user_means[u]
                ptr += nnz_u
        R_csr = R_csr_float
        R_csc = R_csr.tocsc()

        print('[Ref-SLIM-scipy] Вычисление косинусного сходства...')
        norms = np.sqrt(R_csr.multiply(R_csr).sum(axis=0)).A1
        norms[norms == 0] = 1
        R_normed = R_csr.copy().multiply(1.0 / norms)
        sim_matrix = (R_normed.T @ R_normed).toarray()
        np.fill_diagonal(sim_matrix, 0)

        print(f'[Ref-SLIM-scipy] Обучение {n_items} задач L-BFGS-B...')
        start = time.time()

        for i in range(n_items):
            if i % 100 == 0:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (n_items - i - 1)
                print(f'  item {i}/{n_items} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s')

            r_i = np.array(R_csc[:, i].todense(), dtype=float).flatten()
            neighbors = np.argsort(sim_matrix[i])[::-1][:self.n_neighbors]
            X = R_csr[:, neighbors].astype(float).toarray()

            XtX = X.T @ X + self.l2_reg * np.eye(self.n_neighbors)
            Xtr = X.T @ r_i

            def objective(w):
                return 0.5 * w @ XtX @ w - Xtr @ w + self.l1_reg * np.sum(np.abs(w))

            def gradient(w):
                return XtX @ w - Xtr

            w0 = np.zeros(self.n_neighbors)
            res = minimize(objective, w0, jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-6})
            w = np.maximum(res.x, 0)
            self.W[i, neighbors] = w

        elapsed = time.time() - start
        print(f'[Ref-SLIM-scipy] Обучение завершено за {elapsed:.1f}s')
        return self

    def predict(self, R):
        R_csr = R.tocsr().astype(float)
        ptr = 0
        for u in range(R_csr.shape[0]):
            nnz_u = R_csr[u].nnz
            if nnz_u > 0:
                R_csr.data[ptr:ptr + nnz_u] -= self.user_means[u]
                ptr += nnz_u
        preds = R_csr @ self.W.T + self.user_means.reshape(-1, 1)
        return np.clip(preds, 1.0, 5.0)

    def predict_user(self, user_idx, R=None):
        if R is None:
            raise ValueError("Передайте матрицу R")
        r_u = np.array(R[user_idx].todense(), dtype=float).flatten()
        r_u_centered = r_u.copy()
        train_items = R[user_idx].nonzero()[1]
        r_u_centered[train_items] -= self.user_means[user_idx]
        preds = r_u_centered @ self.W.T + self.user_means[user_idx]
        return np.clip(preds, 1.0, 5.0)


def evaluate(model, R_train, R_test, name=''):
    """RMSE на тесте + NDCG@10."""
    R_pred = model.predict(R_train)

    # RMSE только на тестовых
    test_coo = R_test.tocoo()
    y_true = test_coo.data.astype(float)
    y_pred = np.array([R_pred[u, i] for u, i in zip(test_coo.row, test_coo.col)])

    test_rmse = rmse(y_true, y_pred)

    # NDCG@10 на выборке 500 пользователей
    sample = np.random.choice(R_test.shape[0], size=min(500, R_test.shape[0]), replace=False)
    ndcg = compute_ndcg(model, R_train, R_test, k=10, sample_users=sample)

    print(f'  [{name}] Test RMSE = {test_rmse:.4f} | NDCG@10 = {ndcg:.4f}')
    results = {'name': name, 'rmse': round(test_rmse, 4), 'ndcg': round(ndcg, 4)}

    return results


def plot_comparison(results_all, output_path):
    """Столбчатая диаграмма сравнения моделей"""
    models = [r['name'] for r in results_all]
    rmses = [r['rmse'] for r in results_all]
    ndcgs = [r['ndcg'] for r in results_all]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors_rmse = ['#2196F3', '#FF9800', '#2196F3', '#FF9800']
    colors_ndcg = ['#4CAF50', '#F44336', '#4CAF50', '#F44336']

    # RMSE меньше=лучше
    bars1 = axes[0].bar(models, rmses, color=colors_rmse, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('RMSE', fontsize=13)
    axes[0].set_title('Test RMSE (less is better)', fontsize=14, fontweight='bold')
    axes[0].set_ylim(max(0, min(rmses) - 0.15), max(rmses) + 0.05)
    for bar, val in zip(bars1, rmses):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}', ha='center',
                     va='bottom', fontsize=11, fontweight='bold')

    # NDCG больше=лучше
    bars2 = axes[1].bar(models, ndcgs, color=colors_ndcg, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('NDCG@10', fontsize=13)
    axes[1].set_title('NDCG@10 (higher is better)', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, max(ndcgs) + 0.1)
    for bar, val in zip(bars2, ndcgs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}', ha='center',
                     va='bottom', fontsize=11, fontweight='bold')

    legend_elements = [Patch(facecolor='#2196F3', edgecolor='black', label='Custom'),
                       Patch(facecolor='#FF9800', edgecolor='black', label='Reference')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curve(losses, name, output_path):
    """Кривая обучения"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses)+1), losses, 'o-', color='#2196F3', markersize=5, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Train RMSE', fontsize=13)
    ax.set_title(f'{name}: Learning Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    np.random.seed(42)

    data_path = download_movielens()
    df = load_movielens(data_path)
    n_users = df['user_id'].max() + 1
    n_items = df['item_id'].max() + 1
    print(f'\n[DATA] Пользователей: {n_users}, Объектов: {n_items}, Рейтингов: {len(df)}')
    print(f'[DATA] Шкала рейтингов: {df["rating"].min()} - {df["rating"].max()}')
    print(f'[DATA] Средний рейтинг: {df["rating"].mean():.2f}')
    print(f'[DATA] Плотность: {len(df) / (n_users * n_items) * 100:.2f}%')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    R_train = build_sparse_matrix(train_df, n_users, n_items)
    R_test = build_sparse_matrix(test_df, n_users, n_items)
    print(f'Train: {R_train.nnz} рейтингов, Test: {R_test.nnz} рейтингов\n')

    all_results = []

    print('=' * 60)
    print('SLIM')
    print('=' * 60)

    print('\n--- Моя реализация SLIM (sklearn ElasticNet, L1+L2) ---')
    my_slim = MySLIM(alpha=0.01, l1_ratio=0.5, n_neighbors=50)
    my_slim.fit(R_train)
    r1 = evaluate(my_slim, R_train, R_test, name='SLIM (custom)')
    all_results.append(r1)

    print('\n--- Эталон SLIM (scipy.optimize, L-BFGS-B) ---')
    ref_slim = ReferenceSLIM_Scipy(alpha=0.01, l1_reg=0.01, l2_reg=0.01, n_neighbors=50)
    ref_slim.fit(R_train)
    r2 = evaluate(ref_slim, R_train, R_test, name='SLIM (ref-scipy)')
    all_results.append(r2)
    ref_slim_available = True

    print('\n' + '=' * 60)
    print('Latent Factor Model - ALS')
    print('=' * 60)

    print('\n--- Моя реализация ALS ---')
    n_factors = 20
    my_als = MyALS(n_factors=n_factors, lambda_reg=0.1, mu_reg=0.1, n_iter=15)
    my_als.fit(R_train)
    r3 = evaluate(my_als, R_train, R_test, name='ALS (custom)')
    all_results.append(r3)

    print('\n--- Эталон Truncated SVD (scipy) ---')
    ref_svd = ReferenceSVD(n_factors=n_factors)
    ref_svd.fit(R_train)
    r4 = evaluate(ref_svd, R_train, R_test, name='SVD (ref-scipy)')
    all_results.append(r4)

    print('\n' + '=' * 60)
    print('ИТОГОВЫЕ РЕЗУЛЬТАТЫ')
    print('=' * 60)
    print(f'{"Model":<25} {"RMSE":>8} {"NDCG@10":>10}')
    print('-' * 45)
    for r in all_results:
        print(f'{r["name"]:<25} {r["rmse"]:>8.4f} {r["ndcg"]:>10.4f}')

    with open(os.path.join(OUT_DIR, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    plot_comparison(all_results, os.path.join(PLOTS_DIR, 'comparison.png'))

    if my_als.train_rmse_history:
        plot_learning_curve(my_als.train_rmse_history, 'ALS (custom)', os.path.join(PLOTS_DIR, 'als_learning_curve.png'))

    k_values = [5, 10, 15, 20]
    ndcg_curves = {}
    models_for_ndcg = [(my_slim, 'SLIM'), (my_als, 'ALS'), (ref_svd, 'SVD(ref)')]
    if ref_slim_available:
        models_for_ndcg.append((ref_slim, 'SLIM(ref)'))
    for model_obj, model_name in models_for_ndcg:
        ndcgs_k = []
        sample = np.random.choice(n_users, size=min(300, n_users), replace=False)
        for k in k_values:
            val = compute_ndcg(model_obj, R_train, R_test, k=k, sample_users=sample)
            ndcgs_k.append(val)
        ndcg_curves[model_name] = ndcgs_k

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for idx, (name, vals) in enumerate(ndcg_curves.items()):
        ax.plot(k_values, vals, f'{markers[idx]}-', label=name, color=colors[idx], markersize=8, linewidth=2)
    ax.set_xlabel('k', fontsize=13)
    ax.set_ylabel('NDCG@k', fontsize=13)
    ax.set_title('NDCG@k: SLIM vs ALS vs SVD', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'ndcg_at_k.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    subset = R_train[:100, :100].toarray()
    mask = subset > 0
    ax.imshow(mask, cmap='Blues', interpolation='nearest', aspect='auto')
    ax.set_xlabel('Items (first 100)', fontsize=13)
    ax.set_ylabel('Users (first 100)', fontsize=13)
    ax.set_title('Sparsity Pattern of Train Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sparsity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df['rating'], bins=np.arange(0.5, 6, 1), color='#2196F3', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Rating', fontsize=13)
    axes[0].set_ylabel('Count', fontsize=13)
    axes[0].set_title('Rating Distribution', fontsize=14, fontweight='bold')

    user_counts = df.groupby('user_id').size()
    axes[1].hist(user_counts, bins=30, color='#4CAF50', edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Ratings per user', fontsize=13)
    axes[1].set_ylabel('Count', fontsize=13)
    axes[1].set_title('Activity Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    W_display = my_slim.W.copy()
    W_abs = np.abs(W_display)
    W_abs[W_abs < 1e-6] = 0
    im = ax.imshow(W_abs[:50, :50], cmap='YlOrRd', interpolation='nearest', aspect='auto',
                   vmin=0, vmax=np.percentile(W_abs[W_abs > 0], 95) if (W_abs > 0).any() else 0.1)
    ax.set_xlabel('Item index', fontsize=13)
    ax.set_ylabel('Item index', fontsize=13)
    ax.set_title('SLIM Weight Matrix (|W|, first 50x50)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='|Weight|')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'slim_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return all_results


if __name__ == '__main__':
    results = main()
