import time
import pandas as pd

from models.gmm import log_likelihood

pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)


def train_eval_model(model, X_train, X_test, log_prefix: str = " ", **train_kwargs):
    start_fit_time = time.time()
    model.fit(X_train, **train_kwargs)
    print(f"Model trained in {time.time() - start_fit_time:.3f} sec")
    print("Evaluation:")
    ll = log_likelihood(X_test, model.means_, model.covariances_, model.weights_)
    print(f"{log_prefix}Log-likelihood: {ll:.2f}")
    return model

