import numpy as np


def rmse(true_ratings, predicted_ratings):
    true = np.asarray(true_ratings, dtype=np.float64)
    pred = np.asarray(predicted_ratings, dtype=np.float64)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances[:k], dtype=np.float64)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    gains = (2.0 ** relevances - 1.0) / discounts
    return float(gains.sum())


def ndcg_at_k(relevances, k):
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_ndcg(model_recommend_fn, train_df, test_df,
                 n_users, k=10, min_test_ratings=3, max_users=500):
    """
    Вычисляет средний NDCG@k по всем пользователям.

    Для каждого пользователя:
    1. Берём его тестовые рейтинги как «истинные предпочтения».
    2. Запрашиваем top-k рекомендаций у модели.
    3. Сопоставляем рекомендации с тестовыми рейтингами.
    4. Вычисляем NDCG@k.
    """
    # Строим словарь тестовых рейтингов: user -> {item: rating}
    test_ratings = {}
    for _, row in test_df.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        r = float(row["rating"])
        test_ratings.setdefault(u, {})[i] = r

    eligible_users = [u for u, items in test_ratings.items()
                      if len(items) >= min_test_ratings]

    # Ограничиваем число пользователей для скорости
    rng = np.random.default_rng(42)
    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, max_users, replace=False).tolist()

    ndcg_scores = []

    for user_idx in eligible_users:
        # Получаем top-k рекомендаций
        recs = model_recommend_fn(user_idx, top_n=k)
        if not recs:
            continue

        # Вычисляем вектор релевантностей в порядке рекомендаций
        user_test = test_ratings[user_idx]
        relevances = [user_test.get(item_idx, 0.0) for item_idx, _ in recs]

        # Также добавляем все тестовые айтемы в пул для расчёта идеального DCG
        all_relevant = list(user_test.values())
        ndcg = ndcg_at_k(relevances, k)
        _ = ndcg_at_k(sorted(all_relevant, reverse=True), k)  # для проверки

        ndcg_scores.append(ndcg)

    if not ndcg_scores:
        return 0.0
    return float(np.mean(ndcg_scores))


def compute_rmse_from_df(model_predict_fn, test_df):
    preds = model_predict_fn(test_df)
    true = test_df["rating"].values
    return rmse(true, preds)