import numpy as np
from surprise import SVD, Dataset, Reader
from scipy.sparse import issparse

def build_surprise_trainset(train_df, rating_scale=(1, 10)):
    reader = Reader(rating_scale=rating_scale)
    surprise_df = train_df[["user_idx", "item_idx", "rating"]].copy()
    surprise_df.columns = ["userID", "itemID", "rating"]
    dataset = Dataset.load_from_df(surprise_df, reader)
    return dataset.build_full_trainset()


class SurpriseSVD:
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005,
                 reg_all=0.02, random_state=42):
        self.algo = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state,
            verbose=False,
        )

    def fit(self, train_df, rating_scale=(1, 10)):
        trainset = build_surprise_trainset(train_df, rating_scale)
        self.algo.fit(trainset)
        return self

    def predict(self, test_df):
        preds = []
        for _, row in test_df.iterrows():
            pred = self.algo.predict(int(row["user_idx"]), int(row["item_idx"]))
            preds.append(pred.est)
        return np.array(preds, dtype=np.float32)

    def recommend(self, user_idx, all_item_indices, seen_items, top_n=10):
        scores = []
        for i in all_item_indices:
            if i in seen_items:
                continue
            pred = self.algo.predict(user_idx, i)
            scores.append((i, pred.est))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]



class KarypisSLIM:

    def __init__(self, l1r=1.0, l2r=1.0, optTol=1e-7, niters=10000):
        from SLIM import SLIM as SLIMModel, SLIMatrix
        self.params = {
            "l1r": l1r,
            "l2r": l2r,
            "optTol": optTol,
            "niters": niters,
        }
        self._model = None
        self._W = None
        self._train_R = None
        self.n_users = None
        self.n_items = None

    def fit(self, train_R):
        from SLIM import SLIM as SLIMModel, SLIMatrix

        self.n_users, self.n_items = train_R.shape
        self._train_R = train_R

        train_slim = SLIMatrix(train_R)
        self._model = SLIMModel()
        self._model.train(self.params, train_slim)

        self._W = self._model.to_csr()
        return self

    def predict(self, test_df, train_R):
        if self._W is None:
            raise RuntimeError("Модель не обучена.")

        if issparse(train_R):
            R = train_R.toarray().astype(np.float32)
        else:
            R = np.array(train_R, dtype=np.float32)

        preds = []
        for _, row in test_df.iterrows():
            u = int(row["user_idx"])
            i = int(row["item_idx"])
            # r̂_{u,i} = сумма_j r_{u,j} * w_{j,i}
            score = float(self._W[i].dot(R[u]))
            preds.append(score)
        return np.array(preds, dtype=np.float32)

    def recommend(self, user_idx, train_R, top_n=10, exclude_seen=True):
        from SLIM import SLIM as SLIMModel, SLIMatrix

        if issparse(train_R):
            r_u = train_R[user_idx]
        else:
            from scipy.sparse import csr_matrix
            r_u = csr_matrix(train_R[[user_idx]])

        user_slim = SLIMatrix(r_u)
        item_ids, scores = self._model.predict(
            user_slim, nrcmds=top_n + (r_u.nnz if exclude_seen else 0),
            returnscores=True
        )
        items = item_ids[0].tolist()
        scrs = scores[0].tolist()

        if exclude_seen:
            seen = set(r_u.nonzero()[1].tolist())
            result = [(i, s) for i, s in zip(items, scrs) if i not in seen]
        else:
            result = list(zip(items, scrs))

        return result[:top_n]