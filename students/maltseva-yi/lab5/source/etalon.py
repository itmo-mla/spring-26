import numpy as np
from tqdm import tqdm

def eval_etalon_slim(train_matrix, test_df, n_items, n_users):
    from SLIM import SLIM, SLIMatrix
    slim_data = SLIMatrix(train_matrix)
    slim_etalon = SLIM()
    params = {'l1_reg': 0.005, 'l2_reg': 0.495}
    slim_etalon.train(params, slim_data)

    rec_ids, rec_scores = slim_etalon.predict(slim_data,
                                              nrcmds=n_items,
                                              returnscores=True)
    user_scores = []
    for u in range(n_users):
        user_scores.append(dict(zip(rec_ids[u], rec_scores[u])))

    y_true_et, y_pred_et = [], []
    for row in tqdm(test_df.itertuples(), total=len(test_df),
                    desc="Эталонный SLIM предсказание"):
        u, i, r = int(row.user_id), int(row.item_id), row.rating
        pred = user_scores[u].get(i, 0.0)
        y_true_et.append(r)
        y_pred_et.append(pred)

    y_true_et = np.array(y_true_et)
    y_pred_et = np.array(y_pred_et)
    rmse = np.sqrt(np.mean((y_true_et - y_pred_et) ** 2))
    return rmse, user_scores

def eval_etalon_als(train_matrix, test_df, n_users, n_items):
    import implicit
    from implicit.als import AlternatingLeastSquares

    alpha = 1.0
    train_confidence = train_matrix.copy().astype(np.float32)
    train_confidence.data = 1.0 + alpha * train_confidence.data
    item_user_conf = train_confidence.T.tocsr()

    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=10,
                                    calculate_training_loss=True, use_gpu=False)
    model.fit(item_user_conf)

    if model.user_factors.shape[0] == n_users:
        U_imp = model.user_factors
        V_imp = model.item_factors
    else:
        U_imp = model.item_factors
        V_imp = model.user_factors

    y_true_imp, y_pred_imp = [], []
    for row in test_df.itertuples():
        u, i, r = int(row.user_id), int(row.item_id), row.rating
        score = np.dot(U_imp[u], V_imp[i])
        pred_rating = (score - 1) / alpha
        y_true_imp.append(r)
        y_pred_imp.append(pred_rating)

    y_true_imp = np.array(y_true_imp)
    y_pred_imp = np.array(y_pred_imp)
    rmse = np.sqrt(np.mean((y_true_imp - y_pred_imp) ** 2))
    return rmse, U_imp, V_imp