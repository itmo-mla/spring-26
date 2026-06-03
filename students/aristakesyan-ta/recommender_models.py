import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNet
from sklearn.metrics import ndcg_score


def build_user_item_matrix(data, user_col, item_col, rating_col):
    users = data[user_col].astype("category")
    items = data[item_col].astype("category")

    matrix = coo_matrix(
        (data[rating_col].astype(float), (users.cat.codes, items.cat.codes)),
        shape=(users.cat.categories.size, items.cat.categories.size),
    ).tocsr()

    return matrix, users.cat.categories, items.cat.categories


def train_test_split_by_user(matrix, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    matrix = matrix.tocsr()
    train = matrix.copy().tolil()
    test = csr_matrix(matrix.shape).tolil()

    for user_idx in range(matrix.shape[0]):
        start, end = matrix.indptr[user_idx], matrix.indptr[user_idx + 1]
        item_indices = matrix.indices[start:end]
        ratings = matrix.data[start:end]

        if item_indices.size < 2:
            continue

        n_test = max(1, int(round(item_indices.size * test_size)))
        n_test = min(n_test, item_indices.size - 1)
        test_positions = rng.choice(item_indices.size, size=n_test, replace=False)

        for position in test_positions:
            item_idx = item_indices[position]
            test[user_idx, item_idx] = ratings[position]
            train[user_idx, item_idx] = 0.0

    train = train.tocsr()
    test = test.tocsr()
    train.eliminate_zeros()
    test.eliminate_zeros()
    return train, test


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_on_matrix(test_matrix, prediction_matrix):
    test = test_matrix.tocoo()
    predictions = prediction_matrix[test.row, test.col]
    return rmse(test.data, predictions)


def ndcg_at_k(test_matrix, score_matrix, train_matrix=None, k=10, relevance_threshold=None):
    test_matrix = test_matrix.tocsr()
    scores = np.array(score_matrix, copy=True)

    if train_matrix is not None:
        train = train_matrix.tocsr()
        train_rows, train_cols = train.nonzero()
        scores[train_rows, train_cols] = -np.inf

    values = []
    for user_idx in range(test_matrix.shape[0]):
        start, end = test_matrix.indptr[user_idx], test_matrix.indptr[user_idx + 1]
        relevant_items = test_matrix.indices[start:end]
        relevance = test_matrix.data[start:end]

        if relevance_threshold is not None:
            relevant_mask = relevance >= relevance_threshold
            relevant_items = relevant_items[relevant_mask]
            relevance = np.ones(np.count_nonzero(relevant_mask))

        if relevant_items.size == 0:
            continue

        user_scores = scores[user_idx]
        top_k = min(k, user_scores.size)
        top_items = np.argpartition(-user_scores, kth=top_k - 1)[:top_k]
        top_items = top_items[np.argsort(-user_scores[top_items])]
        relevance_by_item = dict(zip(relevant_items, relevance))

        dcg = 0.0
        for rank, item_idx in enumerate(top_items, start=1):
            rel = relevance_by_item.get(item_idx, 0.0)
            dcg += (2.0**rel - 1.0) / np.log2(rank + 1)

        ideal = np.sort(relevance)[::-1][:k]
        idcg = sum((2.0**rel - 1.0) / np.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
        if idcg > 0:
            values.append(dcg / idcg)

    return float(np.mean(values)) if values else 0.0


def ndcg_on_test_items(test_matrix, score_matrix, k=10):
    test_matrix = test_matrix.tocsr()
    values = []

    for user_idx in range(test_matrix.shape[0]):
        start, end = test_matrix.indptr[user_idx], test_matrix.indptr[user_idx + 1]
        item_indices = test_matrix.indices[start:end]
        relevance = test_matrix.data[start:end]

        if item_indices.size == 0:
            continue

        scores = score_matrix[user_idx, item_indices]
        values.append(
            ndcg_score(
                relevance.reshape(1, -1),
                scores.reshape(1, -1),
                k=min(k, item_indices.size),
            )
        )

    return float(np.mean(values)) if values else 0.0


class SLIMRecommender:
    def __init__(
        self,
        alpha=0.01,
        l1_ratio=0.001,
        positive_only=True,
        max_iter=1000,
        tol=1e-4,
        min_rating=1.0,
        max_rating=5.0,
        observed_only=False,
        clip_predictions=False,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.max_iter = max_iter
        self.tol = tol
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.observed_only = observed_only
        self.clip_predictions = clip_predictions
        self.weights_ = None
        self.prediction_matrix_ = None

    def fit(self, train_matrix):
        ratings = self._to_dense(train_matrix)
        n_items = ratings.shape[1]
        weights = np.zeros((n_items, n_items), dtype=float)
        observed_ratings = ratings[ratings > 0]
        self.global_mean_ = float(observed_ratings.mean())

        for item_idx in range(n_items):
            target = ratings[:, item_idx]
            observed_mask = target > 0
            features = ratings.copy()
            features[:, item_idx] = 0.0

            if np.count_nonzero(observed_mask) == 0:
                continue

            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=False,
                positive=self.positive_only,
                max_iter=self.max_iter,
                tol=self.tol,
                selection="cyclic",
            )
            if self.observed_only:
                model.fit(features[observed_mask], target[observed_mask])
            else:
                model.fit(features, target)
            weights[:, item_idx] = model.coef_
            weights[item_idx, item_idx] = 0.0

        self.weights_ = weights
        predictions = ratings @ weights
        if self.observed_only:
            predictions[predictions <= 0] = self.global_mean_
        if self.clip_predictions:
            predictions = np.clip(predictions, self.min_rating, self.max_rating)
        self.prediction_matrix_ = predictions
        return self

    def predict(self, user_indices, item_indices):
        self._check_is_fitted()
        return self.prediction_matrix_[user_indices, item_indices]

    def _check_is_fitted(self):
        if self.weights_ is None or self.prediction_matrix_ is None:
            raise ValueError("Model is not fitted yet.")

    @staticmethod
    def _to_dense(matrix):
        return matrix.toarray().astype(float) if issparse(matrix) else np.asarray(matrix, dtype=float)


class FunkSVDRecommender:
    def __init__(
        self,
        n_factors=32,
        n_epochs=20,
        learning_rate=0.01,
        reg=0.05,
        random_state=42,
        min_rating=1.0,
        max_rating=5.0,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.random_state = random_state
        self.min_rating = min_rating
        self.max_rating = max_rating

    def fit(self, user_indices, item_indices, ratings, n_users, n_items):
        rng = np.random.default_rng(self.random_state)
        self.global_mean_ = float(np.mean(ratings))
        self.user_bias_ = np.zeros(n_users)
        self.item_bias_ = np.zeros(n_items)
        self.user_factors_ = 0.1 * rng.standard_normal((n_users, self.n_factors))
        self.item_factors_ = 0.1 * rng.standard_normal((n_items, self.n_factors))

        indices = np.arange(ratings.size)
        for _ in range(self.n_epochs):
            rng.shuffle(indices)
            for idx in indices:
                user_idx = user_indices[idx]
                item_idx = item_indices[idx]
                rating = ratings[idx]
                prediction = self._raw_predict(user_idx, item_idx)
                error = rating - prediction

                user_vector = self.user_factors_[user_idx].copy()
                item_vector = self.item_factors_[item_idx].copy()

                self.user_bias_[user_idx] += self.learning_rate * (
                    error - self.reg * self.user_bias_[user_idx]
                )
                self.item_bias_[item_idx] += self.learning_rate * (
                    error - self.reg * self.item_bias_[item_idx]
                )
                self.user_factors_[user_idx] += self.learning_rate * (
                    error * item_vector - self.reg * user_vector
                )
                self.item_factors_[item_idx] += self.learning_rate * (
                    error * user_vector - self.reg * item_vector
                )

        return self

    def predict(self, user_indices, item_indices):
        predictions = np.array(
            [self._raw_predict(user_idx, item_idx) for user_idx, item_idx in zip(user_indices, item_indices)]
        )
        return np.clip(predictions, self.min_rating, self.max_rating)

    def predict_all(self):
        predictions = (
            self.global_mean_
            + self.user_bias_[:, np.newaxis]
            + self.item_bias_[np.newaxis, :]
            + self.user_factors_ @ self.item_factors_.T
        )
        return np.clip(predictions, self.min_rating, self.max_rating)

    def _raw_predict(self, user_idx, item_idx):
        return (
            self.global_mean_
            + self.user_bias_[user_idx]
            + self.item_bias_[item_idx]
            + self.user_factors_[user_idx] @ self.item_factors_[item_idx]
        )


class TruncatedSVDRecommender:
    def __init__(self, n_components=32, random_state=42, min_rating=1.0, max_rating=5.0):
        self.n_components = n_components
        self.random_state = random_state
        self.min_rating = min_rating
        self.max_rating = max_rating

    def fit(self, train_matrix):
        train_matrix = train_matrix.tocsr()
        self.global_mean_ = float(train_matrix.data.mean())
        centered = train_matrix.copy()
        centered.data = centered.data - self.global_mean_

        self.model_ = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        user_factors = self.model_.fit_transform(centered)
        predictions = user_factors @ self.model_.components_ + self.global_mean_
        self.prediction_matrix_ = np.clip(predictions, self.min_rating, self.max_rating)
        return self

    def predict(self, user_indices, item_indices):
        return self.prediction_matrix_[user_indices, item_indices]
