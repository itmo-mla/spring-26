import numpy as np
from scipy.sparse import find, csr_matrix


def rmse(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def ndcg_at_k(y_true, y_pred, k=10):
    mask = y_true != 0
    if not mask.any():
        return 0.0
    
    relevant = y_true[mask]
    scores = y_pred[mask]
    
    idx = np.argsort(scores)[::-1][:k]
    rel_sorted = relevant[idx]
    
    if len(rel_sorted) == 0:
        return 0.0
    
    dcg = np.sum((2 ** rel_sorted - 1) / np.log2(np.arange(2, len(rel_sorted) + 2)))
    ideal = np.sort(relevant)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_all(R_true, R_pred, test_mask):
    """
    R_true: исходная матрица оценок (sparse)
    R_pred: предсказанная матрица (dense или sparse)
    test_mask: маска тестовых элементов (bool numpy array)
    """
    # Приводим R_pred к правильной форме если нужно
    if hasattr(R_pred, 'toarray'):
        R_pred_dense = R_pred.toarray()
    else:
        R_pred_dense = np.array(R_pred)
    
    # Приводим форму если нужно
    if R_pred_dense.shape != R_true.shape:
        if R_pred_dense.shape[1] != R_true.shape[1]:
            # Обрезаем или дополняем
            if R_pred_dense.shape[1] < R_true.shape[1]:
                # Дополняем нулями
                pad = np.zeros((R_pred_dense.shape[0], R_true.shape[1] - R_pred_dense.shape[1]))
                R_pred_dense = np.hstack([R_pred_dense, pad])
            else:
                # Обрезаем
                R_pred_dense = R_pred_dense[:, :R_true.shape[1]]
    
    R_true_dense = R_true.toarray()
    
    # Получаем только тестовые значения
    test_true = R_true_dense[test_mask]
    test_pred = R_pred_dense[test_mask]
    
    overall_rmse = np.sqrt(np.mean((test_true - test_pred) ** 2))
    
    # NDCG@10 по пользователям
    ndcg_scores = []
    n_users = R_true.shape[0]
    
    for u in range(n_users):
        user_test_mask = test_mask[u]
        if user_test_mask.sum() < 2:
            continue
        
        y_true_u = R_true_dense[u, user_test_mask]
        y_pred_u = R_pred_dense[u, user_test_mask]
        
        ndcg_scores.append(ndcg_at_k(y_true_u, y_pred_u, k=min(10, len(y_true_u))))
    
    return {
        'RMSE': overall_rmse,
        'NDCG@10': np.mean(ndcg_scores) if ndcg_scores else 0
    }
