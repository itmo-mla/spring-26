import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def cross_validate(model_fn, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, times = [], [], []

    for fold, (tr, val) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_val = scaler.transform(X_val)

        model = model_fn()
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0

        y_pred = model.predict(X_val)
        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, average="macro"))
        times.append(elapsed)
        print(f"  Fold {fold}: acc={accs[-1]:.4f}  f1={f1s[-1]:.4f}  t={elapsed:.1f}s")

    return {
        "accuracy": np.array(accs),
        "f1": np.array(f1s),
        "time": np.array(times),
    }
