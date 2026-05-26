from dataclasses import dataclass
import numpy as np
from scipy import sparse


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    diff = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def ndcg_at_k(
    relevances: np.ndarray,
    scores: np.ndarray,
    k: int = 10,
) -> float:
    relevances = np.asarray(relevances, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if relevances.size == 0:
        return 0.0

    k = min(k, relevances.size)
    order = np.argsort(-scores)[:k]
    gains = relevances[order]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains / discounts))

    ideal = np.sort(relevances)[::-1][:k]
    idcg = float(np.sum(ideal / discounts))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mean_ndcg(
    test_pairs: list[tuple[int, int, float]],
    predict_fn,
    user_items: dict[int, list[int]],
    item_scores: dict[tuple[int, int], float],
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> float:
    by_user: dict[int, list[tuple[int, float]]] = {}
    for u, i, r in test_pairs:
        by_user.setdefault(u, []).append((i, r))

    values = []
    for u, pairs in by_user.items():
        items = user_items.get(u)
        if not items:
            items = [i for i, _ in pairs]
        rel = np.array(
            [1.0 if item_scores.get((u, i), r) >= relevance_threshold else 0.0 for i, r in pairs]
            if len(pairs) == len(items)
            else [1.0 if r >= relevance_threshold else 0.0 for _, r in pairs]
        )
        # оценка всех кандидатов для этого пользователя (из теста + тренировочного профиля)
        cand_items = list({i for i, _ in pairs} | set(items))
        rel_map = {i: (1.0 if item_scores.get((u, i), 0) >= relevance_threshold else 0.0) for i in cand_items}
        rel = np.array([rel_map[i] for i in cand_items])
        pred = np.array([predict_fn(u, i) for i in cand_items])
        values.append(ndcg_at_k(rel, pred, k=k))
    return float(np.mean(values)) if values else 0.0


def _soft_threshold(value: float, lam: float) -> float:
    if value > lam:
        return value - lam
    if value < -lam:
        return value + lam
    return 0.0


def _elastic_net_cd(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l1: float,
    l2: float,
    max_iter: int = 200,
    tol: float = 1e-5,
    positive: bool = True,
    exclude_idx: int | None = None,
) -> np.ndarray:
    n_features = X.shape[1]
    w = np.zeros(n_features, dtype=np.float64)
    if exclude_idx is not None:
        w[exclude_idx] = 0.0

    col_norm = np.sum(X * X, axis=0) + l2
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(n_features):
            if exclude_idx is not None and j == exclude_idx:
                w[j] = 0.0
                continue
            residual = y - X @ w + X[:, j] * w[j]
            rho = float(X[:, j] @ residual)
            z = _soft_threshold(rho, l1) / col_norm[j] if col_norm[j] > 0 else 0.0
            if positive:
                z = max(0.0, z)
            w[j] = z
        if np.linalg.norm(w - w_old, ord=np.inf) < tol:
            break
    if exclude_idx is not None:
        w[exclude_idx] = 0.0
    return w


@dataclass
class SLIM:
    l1: float = 1.0
    l2: float = 1.0
    max_iter: int = 200
    positive: bool = True
    top_k: int | None = 100

    W_: sparse.csr_matrix | None = None
    n_users_: int = 0
    n_items_: int = 0

    def fit(self, R: sparse.csr_matrix) -> "SLIM":
        R = R.tocsr()
        self.n_users_, self.n_items_ = R.shape
        X = R.toarray().astype(np.float64)
        rows, cols, data = [], [], []

        for j in range(self.n_items_):
            y = X[:, j]
            if np.count_nonzero(y) < 2:
                continue
            Xj = X.copy()
            Xj[:, j] = 0.0
            w = _elastic_net_cd(
                Xj,
                y,
                l1=self.l1,
                l2=self.l2,
                max_iter=self.max_iter,
                positive=self.positive,
                exclude_idx=j,
            )
            if self.top_k is not None and self.top_k > 0:
                idx = np.argsort(-w)[: self.top_k]
                mask = np.zeros_like(w, dtype=bool)
                mask[idx] = w[idx] > 0
                w = w * mask
            nz = np.flatnonzero(w)
            rows.extend([j] * len(nz))
            cols.extend(nz.tolist())
            data.extend(w[nz].tolist())

        self.W_ = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_items_, self.n_items_),
        )
        return self

    def predict_matrix(self, R: sparse.csr_matrix) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError("Model is not fitted.")
        return (R @ self.W_).toarray()

    def predict(self, user_id: int, item_id: int, R: sparse.csr_matrix) -> float:
        row = R.getrow(user_id).toarray().ravel()
        return float(row @ self.W_[:, item_id].toarray().ravel())


@dataclass
class LSARecommender:

    n_components: int = 50
    max_features: int = 5000

    item_factors_: np.ndarray | None = None
    user_factors_: np.ndarray | None = None
    global_mean_: float = 0.0
    item_index_: dict[str, int] | None = None

    def fit(
        self,
        R: sparse.csr_matrix,
        item_documents: list[str],
        item_ids: list[str],
    ) -> "LSARecommender":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.global_mean_ = float(R.data.mean()) if R.nnz else 0.0
        self.item_index_ = {doc_id: idx for idx, doc_id in enumerate(item_ids)}

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            min_df=2,
        )
        tfidf = vectorizer.fit_transform(item_documents)
        n_comp = min(self.n_components, max(1, tfidf.shape[0] - 1), max(1, tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.item_factors_ = svd.fit_transform(tfidf)

        n_users = R.shape[0]
        n_items = len(item_ids)
        n_comp = self.item_factors_.shape[1]
        user_factors = np.zeros((n_users, n_comp), dtype=np.float64)
        user_counts = np.zeros(n_users, dtype=np.float64)

        R_coo = R.tocoo()
        for u, i, r in zip(R_coo.row, R_coo.col, R_coo.data):
            if i < n_items:
                user_factors[u] += (r - self.global_mean_) * self.item_factors_[i]
                user_counts[u] += 1.0

        for u in range(n_users):
            if user_counts[u] > 0:
                user_factors[u] /= user_counts[u]
        self.user_factors_ = user_factors
        return self

    def predict_matrix(self) -> np.ndarray:
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.global_mean_ + self.user_factors_ @ self.item_factors_.T

    def predict(self, user_id: int, item_id: int) -> float:
        mat = self.predict_matrix()
        return float(mat[user_id, item_id])


def build_user_item_matrix(
    df,
    user_col: str,
    item_col: str,
    rating_col: str,
    *,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
) -> tuple[sparse.csr_matrix, dict, dict]:
    counts_u = df.groupby(user_col).size()
    counts_i = df.groupby(item_col).size()
    df = df[
        df[user_col].isin(counts_u[counts_u >= min_user_ratings].index)
        & df[item_col].isin(counts_i[counts_i >= min_item_ratings].index)
    ].copy()

    users = df[user_col].astype(str).unique()
    items = df[item_col].astype(str).unique()
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}

    rows = df[user_col].astype(str).map(user_to_idx).to_numpy()
    cols = df[item_col].astype(str).map(item_to_idx).to_numpy()
    data = df[rating_col].astype(np.float64).to_numpy()
    R = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(items)),
    )
    return R, user_to_idx, item_to_idx


def train_test_split_ratings(
    df,
    user_col: str,
    item_col: str,
    rating_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    rng = np.random.default_rng(random_state)
    mask = rng.random(len(df)) < test_size
    train_df = df.loc[~mask].copy()
    test_df = df.loc[mask].copy()
    return train_df, test_df


def predict_pairs(
    R_train: sparse.csr_matrix,
    test_df,
    user_to_idx: dict,
    item_to_idx: dict,
    user_col: str,
    item_col: str,
    rating_col: str,
    model: SLIM | LSARecommender,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    if isinstance(model, SLIM):
        full_pred = model.predict_matrix(R_train)
    else:
        full_pred = model.predict_matrix()
    for _, row in test_df.iterrows():
        u = user_to_idx.get(str(row[user_col]))
        i = item_to_idx.get(str(row[item_col]))
        if u is None or i is None:
            continue
        y_true.append(row[rating_col])
        y_pred.append(full_pred[u, i])
    return np.array(y_true), np.array(y_pred)
