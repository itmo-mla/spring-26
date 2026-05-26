import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def load_movielens_100k():
    ratings = pd.read_csv(
        'http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
        sep='\t', names=['userId', 'movieId', 'rating', 'timestamp']
    )
    movies = pd.read_csv(
        'http://files.grouplens.org/datasets/movielens/ml-100k/u.item',
        sep='|', encoding='latin-1',
        names=['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
               'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy',
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    )
    unique_users = ratings['userId'].unique()
    unique_movies = ratings['movieId'].unique()
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
    ratings['user_idx'] = ratings['userId'].map(user_to_idx)
    ratings['movie_idx'] = ratings['movieId'].map(movie_to_idx)
    n_users = len(unique_users)
    n_items = len(unique_movies)
    matrix = csr_matrix(
        (ratings['rating'].astype(np.float32), (ratings['user_idx'], ratings['movie_idx'])),
        shape=(n_users, n_items)
    )
    print(f"Dataset: {n_users} users, {n_items} items, {matrix.nnz} ratings")
    print(f"Density: {matrix.nnz / (n_users * n_items) * 100:.4f}%")
    return matrix, ratings, user_to_idx, movie_to_idx, movies