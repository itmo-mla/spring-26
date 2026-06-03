import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix

def load_movielens100k():
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    if not os.path.exists("ml-100k"):
        print("Скачивание MovieLens 100k...")
        urllib.request.urlretrieve(url, "ml-100k.zip")
        with zipfile.ZipFile("ml-100k.zip", 'r') as z:
            z.extractall(".")
        os.remove("ml-100k.zip")
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=cols)
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1
    return df[['user_id', 'item_id', 'rating']]