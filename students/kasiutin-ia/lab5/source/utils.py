import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

MOVIELENS_RATINGS_PATH = "ml-latest-small/ratings.csv"

COLS_TO_SELECT = [
    "userId",
    "movieId",
    "rating",
]


def _load_interactions(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, usecols=COLS_TO_SELECT)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    return (
        df.groupby(["userId", "movieId"], as_index=False)["rating"]
        .mean()
        .astype({"userId": int, "movieId": int})
    )


def _interactions_to_matrix(
    interactions: pd.DataFrame,
    user_ids: list[int],
    item_ids: list[int],
) -> csr_matrix:
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    rows = interactions["userId"].map(user_index).to_numpy()
    cols = interactions["movieId"].map(item_index).to_numpy()
    ratings = interactions["rating"].to_numpy(dtype=np.float32)

    return csr_matrix(
        (ratings, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )


def build_interaction_matrix(
    input_path: str = MOVIELENS_RATINGS_PATH,
) -> tuple[csr_matrix, list[int], list[int]]:
    """
    Строит матрицу взаимодействий user × item из MovieLens ratings.csv.

    Значения ячеек — оценки (rating). При дубликатах пар (user, movie)
    берётся среднее.
    """
    interactions = _load_interactions(input_path)
    user_ids = sorted(interactions["userId"].unique())
    item_ids = sorted(interactions["movieId"].unique())

    R = _interactions_to_matrix(interactions, user_ids, item_ids)
    return R, user_ids, item_ids


def train_test_split_ratings(
    input_path: str = MOVIELENS_RATINGS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    csr_matrix,
    list[int],
    list[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Разбивает оценки на train и test с общей индексацией пользователей и фильмов.

    Returns
    -------
    R_train : csr_matrix
        Обучающая матрица user × item.
    user_ids, item_ids : list[int]
        Соответствие индексов матрицы исходным id.
    test_user_idx, test_item_idx : np.ndarray
        Индексы строк/столбцов для тестовых пар.
    test_ratings : np.ndarray
        Истинные оценки в test.
    """
    interactions = _load_interactions(input_path)
    user_ids = sorted(interactions["userId"].unique())
    item_ids = sorted(interactions["movieId"].unique())
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    train_df, test_df = train_test_split(
        interactions,
        test_size=test_size,
        random_state=random_state,
    )

    R_train = _interactions_to_matrix(train_df, user_ids, item_ids)

    test_user_idx = test_df["userId"].map(user_index).to_numpy()
    test_item_idx = test_df["movieId"].map(item_index).to_numpy()
    test_ratings = test_df["rating"].to_numpy(dtype=np.float32)

    return R_train, user_ids, item_ids, test_user_idx, test_item_idx, test_ratings


def _is_model_fitted(model) -> bool:
    if getattr(model, "P", None) is not None and getattr(model, "Q", None) is not None:
        return True
    if getattr(model, "W", None) is not None:
        return True
    return False


def rmse_on_test(
    model,
    R_train: csr_matrix,
    test_user_idx: np.ndarray,
    test_item_idx: np.ndarray,
    test_ratings: np.ndarray,
) -> float:
    """
    RMSE на тестовых парах (user, item).

    Для SLIM: R_hat = R_train @ W.
    Для ALS: R_hat = μ + b_u + b_i + P Q^T.
    """
    if not _is_model_fitted(model):
        raise ValueError("Модель не обучена: сначала вызовите fit().")

    R_hat = model.predict(R_train)
    preds = np.asarray(R_hat[test_user_idx, test_item_idx]).ravel()
    y_true = np.asarray(test_ratings, dtype=np.float64)

    return float(np.sqrt(np.mean((y_true - preds) ** 2)))


def compare_models_rmse(
    models: dict[str, object],
    R_train: csr_matrix,
    test_user_idx: np.ndarray,
    test_item_idx: np.ndarray,
    test_ratings: np.ndarray,
) -> pd.Series:
    """Считает test RMSE для нескольких моделей с интерфейсом fit/predict."""
    return pd.Series(
        {
            name: rmse_on_test(model, R_train, test_user_idx, test_item_idx, test_ratings)
            for name, model in models.items()
        },
        name="test_rmse",
    )


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    relevances = np.asarray(relevances, dtype=np.float64)[:k]
    if relevances.size == 0:
        return 0.0
    positions = np.arange(1, relevances.size + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg_at_k(
    model,
    R_train: csr_matrix,
    test_user_idx: np.ndarray,
    test_item_idx: np.ndarray,
    test_ratings: np.ndarray,
    k: int = 10,
) -> float:
    """
    Mean NDCG@k по пользователям, у которых есть оценки в test.

    Релевантность = значение рейтинга в test; уже просмотренные в train фильмы
    исключаются из ранжирования.
    """
    if not _is_model_fitted(model):
        raise ValueError("Модель не обучена: сначала вызовите fit().")

    scores_matrix = model.predict(R_train)
    if hasattr(scores_matrix, "toarray"):
        scores_matrix = scores_matrix.toarray()
    else:
        scores_matrix = np.asarray(scores_matrix)
    test_by_user: dict[int, list[tuple[int, float]]] = {}
    for u, i, r in zip(test_user_idx, test_item_idx, test_ratings):
        test_by_user.setdefault(int(u), []).append((int(i), float(r)))

    ndcgs = []
    for user_idx, item_rels in test_by_user.items():
        scores = scores_matrix[user_idx].copy()
        scores[R_train[user_idx].indices] = -np.inf

        top_items = np.argpartition(-scores, k)[:k]
        top_items = top_items[np.argsort(-scores[top_items])]

        rel_map = dict(item_rels)
        gained = np.array([rel_map.get(int(i), 0.0) for i in top_items], dtype=np.float64)
        ideal = np.array(sorted((r for _, r in item_rels), reverse=True)[:k], dtype=np.float64)

        idcg = _dcg_at_k(ideal, k)
        if idcg > 0:
            ndcgs.append(_dcg_at_k(gained, k) / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0
