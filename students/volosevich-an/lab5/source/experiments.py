# %% [markdown]
# # Рекомендательная система для Amazon Video Games
# ## Сравнение SLIM и SVD

# %%
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import find, csr_matrix
import warnings
warnings.filterwarnings('ignore')

from data_downloader import load_or_generate_data
from slim import SLIM
from svd import SVD
from metrics import evaluate_all

# Загрузка данных
R, train_mask, test_mask = load_or_generate_data(max_users=500, max_items=800)
R_train = R.multiply(train_mask)
R_train_csr = csr_matrix(R_train)

print(f"Размер матрицы: {R.shape}")
print(f"Пользователей: {R.shape[0]}, Товаров: {R.shape[1]}")
print(f"Оценок в train: {R_train.nnz}")
print(f"Оценок в test: {R.multiply(test_mask).nnz}")
print(f"Разреженность: {1 - R.nnz / (R.shape[0] * R.shape[1]):.2%}")

# %% [markdown]
# ### 1. Загрузка данных

# %%
R, train_mask, test_mask = load_or_generate_data(max_users=500, max_items=800)
R_train = R.multiply(train_mask)

print(f"Размер матрицы: {R.shape}")
print(f"Оценок в train: {R_train.nnz}")
print(f"Оценок в test: {R.multiply(test_mask).nnz}")
print(f"Разреженность: {1 - R.nnz / (R.shape[0] * R.shape[1]):.2%}")

# %% [markdown]
# ### 2. Обучение SLIM

# %%
slim = SLIM(alpha=0.01, l1_ratio=0.01, top_k=100, max_iter=2000)
slim.fit(R_train_csr)
R_pred_slim = slim.predict(R_train_csr)

# Приводим к правильной форме если нужно
if hasattr(R_pred_slim, 'toarray'):
    R_pred_slim = R_pred_slim.toarray()

metrics_slim = evaluate_all(R, R_pred_slim, test_mask)
print(f"SLIM - RMSE: {metrics_slim['RMSE']:.4f}, NDCG@10: {metrics_slim['NDCG@10']:.4f}")

# %% [markdown]
# ### 3. Обучение SVD

# %%
svd = SVD(n_factors=50, learning_rate=0.005, reg=0.02, n_epochs=30)
svd.fit(R_train_csr)
R_pred_svd = svd.predict(R_train_csr)

if hasattr(R_pred_svd, 'toarray'):
    R_pred_svd = R_pred_svd.toarray()

metrics_svd = evaluate_all(R, R_pred_svd, test_mask)
print(f"Наша SVD - RMSE: {metrics_svd['RMSE']:.4f}, NDCG@10: {metrics_svd['NDCG@10']:.4f}")

# %% [markdown]
# ### 4. Эталон SVD

# %%

try:
    from surprise import SVD as SurpriseSVD, Dataset, Reader
    from surprise.model_selection import train_test_split as surprise_split
    
    # Подготовка данных для surprise
    rows, cols, ratings = find(R)
    df_ratings = pd.DataFrame({
        'user_id': rows,
        'item_id': cols,
        'rating': ratings
    })
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_ratings, reader)
    

    trainset, testset = surprise_split(data, test_size=0.2, random_state=42)
    
    surprise_svd = SurpriseSVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
    surprise_svd.fit(trainset)

    predictions = surprise_svd.test(testset)
    
    # Расчёт RMSE
    from surprise import accuracy
    surprise_rmse = accuracy.rmse(predictions, verbose=False)
    
    # Расчёт NDCG@10
    from collections import defaultdict
    
    # Группируем предсказания по пользователям
    user_predictions = defaultdict(list)
    for pred in predictions:
        user_predictions[pred.uid].append((pred.iid, pred.est, pred.r_ui))
    
    ndcg_scores = []
    for uid, items in user_predictions.items():
        # Сортируем по предсказанию
        items.sort(key=lambda x: x[1], reverse=True)
        top_k = items[:10]
        
        # Релевантность
        rel = [1 if item[2] >= 4 else 0 for item in top_k]
        
        if len(rel) == 0:
            continue
        
        # DCG
        dcg = sum(rel[i] / np.log2(i + 2) for i in range(len(rel)))
        
        # IDCG
        ideal_rel = sorted([1 if item[2] >= 4 else 0 for item in items], reverse=True)[:10]
        idcg = sum(ideal_rel[i] / np.log2(i + 2) for i in range(len(ideal_rel)))
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    surprise_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    
    print(f"Эталонная SVD (surprise) - RMSE: {surprise_rmse:.4f}, NDCG@10: {surprise_ndcg:.4f}")
    
    metrics_surprise = {'RMSE': surprise_rmse, 'NDCG@10': surprise_ndcg}
    
except ImportError:
    print("Библиотека surprise не установлена. Установите: pip install scikit-surprise")
    metrics_surprise = {'RMSE': None, 'NDCG@10': None}

# %% [markdown]
# ### 5. ЭТАЛОН SLIM

# %%
import sys
import os
import numpy as np
from scipy.sparse import find

SLIM_PATH = "/Users/artemvolosevich/Documents/ITMO/Second-Term/spring-26/SLIM"

sys.path.insert(0, f"{SLIM_PATH}/python-package")
os.environ["DYLD_LIBRARY_PATH"] = f"{SLIM_PATH}/build/src/libslim"

from SLIM.core import SLIMatrix
from SLIM import SLIM

rows, cols, ratings = find(R_train_csr)
train_data = np.vstack([rows, cols, ratings]).T.astype(np.float64)
trainmat = SLIMatrix(train_data)

model = SLIM()
params = {
    'algo': 'cd',
    'nthreads': 4,
    'l1r': 0.01,
    'l2r': 0.01,
    'niters': 2000
}
model.train(params, trainmat)

test_rows, test_cols = np.where(test_mask)
test_data = np.vstack([test_rows, test_cols, np.ones(len(test_rows))]).T.astype(np.float64)
testmat = SLIMatrix(test_data, trainmat)
predictions = model.predict(testmat, nrcmds=10)

K = 10
hits = 0
users = 0

for u, recs in predictions.items():
    u = int(u)
    if u < 0 or u >= R_train_csr.shape[0]:
        continue
    true_items = set(R_train_csr[u].indices)
    if len(true_items) == 0:
        continue
    recs = np.asarray(recs, dtype=int)[:K]
    hits += len(set(recs) & true_items)
    users += 1

recall_etalon = hits / users if users > 0 else 0.0
print(f"Эталон SLIM - Recall@10: {recall_etalon:.4f}")

# %% [markdown]
# ### 6. СРАВНЕНИЕ АЛГОРИТМОВ

# %%

import numpy as np
import matplotlib.pyplot as plt

def recall_at_k_simple(R_true, R_pred, test_mask, k=10):
    R_true_dense = R_true.toarray()
    if hasattr(R_pred, 'toarray'):
        R_pred_dense = R_pred.toarray()
    else:
        R_pred_dense = R_pred
    if R_pred_dense.shape[1] != R_true_dense.shape[1]:
        if R_pred_dense.shape[1] > R_true_dense.shape[1]:
            R_pred_dense = R_pred_dense[:, :R_true_dense.shape[1]]
        else:
            pad = np.zeros((R_pred_dense.shape[0], R_true_dense.shape[1] - R_pred_dense.shape[1]))
            R_pred_dense = np.hstack([R_pred_dense, pad])
    hits, users = 0, 0
    for u in range(R_true_dense.shape[0]):
        true = set(np.where((R_true_dense[u] * test_mask[u]) > 0)[0])
        if not true:
            continue
        pred = np.argsort(R_pred_dense[u])[::-1][:k]
        hits += len(set(pred) & true)
        users += 1
    return hits / users if users > 0 else 0.0

results = {}

# Эталон SLIM (recall_at_k - это число из вашей ячейки)
# Убедитесь, что recall_at_k - это float, а не функция
if 'recall_at_k' in globals() and not callable(recall_etalon):
    results['Эталон SLIM'] = {'recall': float(recall_etalon), 'rmse': None}


# Наша SLIM
if 'R_pred_slim' in globals():
    results['Наша SLIM'] = {
        'recall': float(recall_at_k_simple(R, R_pred_slim, test_mask, 10)),
        'rmse': float(metrics_slim['RMSE'])
    }

# Наша SVD
if 'R_pred_svd' in globals():
    results['Наша SVD'] = {
        'recall': float(recall_at_k_simple(R, R_pred_svd, test_mask, 10)),
        'rmse': float(metrics_svd['RMSE'])
    }

# SVD surprise
if 'R_pred_surprise' in globals():
    results['SVD (surprise)'] = {
        'recall': float(recall_at_k_simple(R, R_pred_surprise, test_mask, 10)),
        'rmse': float(metrics_surprise['RMSE'])
    }

# Бейзлайн
item_popularity = np.array(R_train.sum(axis=0)).ravel()
R_pred_pop = np.tile(item_popularity, (R.shape[0], 1))
from metrics import evaluate_all
results['Бейзлайн'] = {
    'recall': float(recall_at_k_simple(R, R_pred_pop, test_mask, 10)),
    'rmse': float(evaluate_all(R, R_pred_pop, test_mask)['RMSE'])
}

print("-" * 45)
print(f"{'Модель':<20} {'RMSE':<10} {'Recall@10':<12}")
print("-" * 45)

for name, m in results.items():
    rmse_val = m['rmse']
    recall_val = m['recall']
    rmse_str = f"{rmse_val:.4f}" if rmse_val is not None else "N/A"
    # Гарантируем, что recall_val - число
    print(f"{name:<20} {rmse_str:<10} {recall_val:.4f}")

# Графики
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
names = list(results.keys())
rmse_vals = [m['rmse'] if m['rmse'] is not None else 0 for m in results.values()]
recall_vals = [m['recall'] for m in results.values()]

axes[0].bar(names, rmse_vals, color='steelblue')
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylabel('RMSE')
axes[0].set_title('RMSE (меньше лучше)')

axes[1].bar(names, recall_vals, color='darkorange')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylabel('Recall@10')
axes[1].set_title('Recall@10 (больше лучше)')

plt.tight_layout()
plt.show()

print(f"\nЛучший Recall@10: {names[np.argmax(recall_vals)]} ({max(recall_vals):.4f})")
print(f"Лучший RMSE: {names[np.argmin(rmse_vals)]} ({min(rmse_vals):.4f})")


