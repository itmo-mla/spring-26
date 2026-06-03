import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import os


def load_amazon_video_games(data_dir='./data', max_users=500, max_items=800, min_ratings_per_user=3):
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, 'video_games.parquet')
    
    if os.path.exists(cache_path):
        print(f"Загрузка из кэша: {cache_path}")
        df = pd.read_parquet(cache_path)
        return df
    
    print("Загрузка Amazon Video Games датасета...")
    print("Это может занять 5-10 минут...")
    
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Video_Games",
        trust_remote_code=True,
        split="full"
    )
    
    print(f"Загружено записей: {len(dataset)}")
    
    data = []
    for i, item in enumerate(dataset):
        if i % 100000 == 0:
            print(f"Обработано {i} записей...")
        data.append({
            'user_id': item['user_id'],
            'item_id': item['parent_asin'],
            'rating': float(item['rating'])
        })
    
    df = pd.DataFrame(data)
    
    # Фильтрация пользователей с малым числом оценок
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    df = df[df['user_id'].isin(valid_users)]
    
    # Оставляем топ пользователей и товаров
    user_counts = df['user_id'].value_counts()
    item_counts = df['item_id'].value_counts()
    
    top_users = user_counts.head(max_users).index
    top_items = item_counts.head(max_items).index
    
    df = df[df['user_id'].isin(top_users) & df['item_id'].isin(top_items)]
    
    # Маппинг в индексы
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {i: j for j, i in enumerate(unique_items)}
    
    df['user_idx'] = df['user_id'].map(user_to_idx)
    df['item_idx'] = df['item_id'].map(item_to_idx)
    
    print(f"Итоговые данные: {len(df)} оценок")
    print(f"Пользователей: {df['user_idx'].nunique()}, Товаров: {df['item_idx'].nunique()}")
    
    df.to_parquet(cache_path, index=False)
    
    return df

def create_matrix(df, n_users, n_items):
    return csr_matrix((df['rating'], (df['user_idx'], df['item_idx'])), shape=(n_users, n_items))

def load_or_generate_data(data_dir='./data', max_users=500, max_items=800):
    os.makedirs(data_dir, exist_ok=True)
    matrix_path = os.path.join(data_dir, 'R_video_games.npz')
    splits_path = os.path.join(data_dir, 'splits_video_games.npz')
    
    if os.path.exists(matrix_path) and os.path.exists(splits_path):
        print("Загрузка сохранённых данных...")
        R = load_npz(matrix_path)
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        splits = np.load(splits_path, allow_pickle=True)
        train_mask = splits['train_mask']
        test_mask = splits['test_mask']
        return R, train_mask, test_mask
    
    df = load_amazon_video_games(data_dir, max_users, max_items)
    
    n_users = df['user_idx'].nunique()
    n_items = df['item_idx'].nunique()
    
    R = create_matrix(df, n_users, n_items)
    
    # Создание train/test масок
    train_mask = np.zeros(R.shape, dtype=bool)
    test_mask = np.zeros(R.shape, dtype=bool)
    
    for u in range(n_users):
        row = R[u].indices
        if len(row) < 2:
            continue
        n_test = max(1, int(len(row) * 0.2))
        test_idx = np.random.choice(row, size=n_test, replace=False)
        test_mask[u, test_idx] = True
    
    train_mask = ~test_mask & (R != 0).toarray()
    
    save_npz(matrix_path, R)
    np.savez(splits_path, train_mask=train_mask, test_mask=test_mask)
    
    print(f"Итоговая матрица: {R.shape}")
    print(f"Разреженность: {1 - R.nnz / (R.shape[0] * R.shape[1]):.2%}")
    print(f"Train оценок: {np.sum(train_mask)}")
    print(f"Test оценок: {np.sum(test_mask)}")
    
    return R, train_mask, test_mask
