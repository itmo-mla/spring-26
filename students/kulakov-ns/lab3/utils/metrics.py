from typing import Dict, Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss


def evaluate_model(name, model, X, y) -> Dict[str, Any]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "logloss": log_loss(y, proba, labels=[0, 1]),
    }
