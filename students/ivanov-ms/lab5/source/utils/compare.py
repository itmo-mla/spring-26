import time
import pandas as pd

from models import BaseRanker
from .metrics import mertics_at_k

pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)


def train_eval_model(model: BaseRanker, X_train, X_train_matrix, X_test, k, log_prefix: str = " "):
    start_fit_time = time.time()
    model.fit(X_train, X_train_matrix)

    print(f"Model trained in {time.time() - start_fit_time:.3f} sec")
    print("Evaluation:")

    topk_preds = model.predict_topk(X_train, X_train_matrix, k=k, explore=k)
    metrics = mertics_at_k(X_test, topk_preds, k=k)

    for metric_name, metric_value in metrics.items():
        print(f"{log_prefix}{metric_name}: {metric_value:.4f}")

    return topk_preds, metrics
