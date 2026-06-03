import numpy as np


class DecisionStump:
    feature_: int
    threshold_: float
    left_val_: float
    right_val_: float

    def fit(self, X: np.ndarray, r: np.ndarray) -> "DecisionStump":
        best_loss = np.inf
        n, m = X.shape

        for f in range(m):
            loss, threshold, lv, rv = self._best_split(X[:, f], r)
            if loss < best_loss:
                best_loss = loss
                self.feature_ = f
                self.threshold_ = threshold
                self.left_val_ = lv
                self.right_val_ = rv

        if best_loss == np.inf:
            self.feature_ = 0
            self.threshold_ = np.inf
            self.left_val_ = r.mean()
            self.right_val_ = r.mean()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mask = X[:, self.feature_] <= self.threshold_
        return np.where(mask, self.left_val_, self.right_val_)

    @staticmethod
    def _best_split(x: np.ndarray, r: np.ndarray):
        order = np.argsort(x)
        xs, rs = x[order], r[order]
        n = len(r)

        cum_r = np.cumsum(rs)
        cum_r2 = np.cumsum(rs ** 2)
        nl = np.arange(1, n)
        nr = n - nl

        sl = cum_r[:-1]
        sl2 = cum_r2[:-1]
        sr = cum_r[-1] - sl
        sr2 = cum_r2[-1] - sl2

        loss = (sl2 - sl ** 2 / nl) + (sr2 - sr ** 2 / nr)

        valid = xs[:-1] < xs[1:]
        if not valid.any():
            return np.inf, None, None, None

        loss = np.where(valid, loss, np.inf)
        i = loss.argmin()
        threshold = (xs[i] + xs[i + 1]) / 2
        return loss[i], threshold, sl[i] / nl[i], sr[i] / nr[i]
