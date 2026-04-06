import numpy as np


def gini(y):
    if len(y) == 0:
        return 0.0
    classes = np.unique(y)
    result = 0.0
    for c in classes:
        p = np.sum(y == c) / len(y)
        result += p * (1 - p)
    return result


def gain(y, y_left, y_right):
    n = len(y)
    if n == 0:
        return 0.0
    return gini(y) - (len(y_left) / n) * gini(y_left) - (len(y_right) / n) * gini(y_right)


def best_split(X, y, cat_features):
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    best_is_cat = False

    n_features = X.shape[1]

    for j in range(n_features):
        col = X[:, j]
        mask = ~np.isnan(col)
        col_valid = col[mask]
        y_valid = y[mask]

        if len(y_valid) < 2:
            continue

        if j in cat_features:
            for k in np.unique(col_valid):
                left = y_valid[col_valid == k]
                right = y_valid[col_valid != k]
                if len(left) == 0 or len(right) == 0:
                    continue
                g = gain(y_valid, left, right)
                if g > best_gain:
                    best_gain = g
                    best_feature = j
                    best_threshold = k
                    best_is_cat = True
        else:
            sorted_vals = np.sort(np.unique(col_valid))
            for i in range(len(sorted_vals) - 1):
                t = (sorted_vals[i] + sorted_vals[i + 1]) / 2
                left = y_valid[col_valid < t]
                right = y_valid[col_valid >= t]
                if len(left) == 0 or len(right) == 0:
                    continue
                g = gain(y_valid, left, right)
                if g > best_gain:
                    best_gain = g
                    best_feature = j
                    best_threshold = t
                    best_is_cat = False

    return best_feature, best_threshold, best_is_cat, best_gain


def major(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]


def class_probs(y, classes):
    probs = {}
    for c in classes:
        probs[c] = np.sum(y == c) / len(y) if len(y) > 0 else 0.0
    return probs


def tree_growing(X, y, cat_features, classes, max_depth=None, min_samples=2, depth=0):
    node = {
        "is_leaf": True,
        "label": major(y),
        "probs": class_probs(y, classes),
        "samples": len(y),
    }

    if len(np.unique(y)) == 1:
        return node
    if len(y) < min_samples:
        return node
    if max_depth is not None and depth >= max_depth:
        return node

    feature, threshold, is_cat, g = best_split(X, y, cat_features)

    if feature is None or g < 1e-7:
        return node

    col = X[:, feature]
    mask = ~np.isnan(col)
    col_valid = col[mask]

    if is_cat:
        left_mask = (col == threshold) & mask
        right_mask = (col != threshold) & mask
    else:
        left_mask = (col < threshold) & mask
        right_mask = (col >= threshold) & mask

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    if len(y_left) == 0 or len(y_right) == 0:
        return node

    n_valid = np.sum(mask)
    q_left = len(y_left) / n_valid
    q_right = len(y_right) / n_valid

    node["is_leaf"] = False
    node["feature"] = feature
    node["threshold"] = threshold
    node["is_cat"] = is_cat
    node["qvk"] = (q_left, q_right)
    node["left"] = tree_growing(X_left, y_left, cat_features, classes, max_depth, min_samples, depth + 1)
    node["right"] = tree_growing(X_right, y_right, cat_features, classes, max_depth, min_samples, depth + 1)

    return node


def predict_proba(node, x, classes):
    if node["is_leaf"]:
        return node["probs"]

    feature = node["feature"]
    threshold = node["threshold"]
    val = x[feature]

    if np.isnan(val):
        q_left, q_right = node["qvk"]
        probs_left = predict_proba(node["left"], x, classes)
        probs_right = predict_proba(node["right"], x, classes)
        result = {}
        for c in classes:
            result[c] = q_left * probs_left.get(c, 0) + q_right * probs_right.get(c, 0)
        return result

    if node["is_cat"]:
        if val == threshold:
            return predict_proba(node["left"], x, classes)
        else:
            return predict_proba(node["right"], x, classes)
    else:
        if val < threshold:
            return predict_proba(node["left"], x, classes)
        else:
            return predict_proba(node["right"], x, classes)


def predict(node, X, classes):
    predictions = []
    for i in range(X.shape[0]):
        probs = predict_proba(node, X[i], classes)
        predictions.append(max(probs, key=probs.get))
    return np.array(predictions)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def prune(node, X_val, y_val, classes):
    if node["is_leaf"]:
        return

    feature = node["feature"]
    threshold = node["threshold"]
    is_cat = node["is_cat"]
    col = X_val[:, feature]
    mask = ~np.isnan(col)

    if is_cat:
        left_mask = (col == threshold) & mask
        right_mask = (col != threshold) & mask
    else:
        left_mask = (col < threshold) & mask
        right_mask = (col >= threshold) & mask

    prune(node["left"], X_val[left_mask], y_val[left_mask], classes)
    prune(node["right"], X_val[right_mask], y_val[right_mask], classes)

    if len(y_val) == 0:
        return

    preds_tree = predict(node, X_val, classes)
    errors_tree = np.sum(preds_tree != y_val)

    errors_leaf = np.sum(y_val != node["label"])

    if errors_leaf <= errors_tree:
        node["is_leaf"] = True
        node.pop("left", None)
        node.pop("right", None)
        node.pop("feature", None)
        node.pop("threshold", None)
        node.pop("is_cat", None)
        node.pop("qvk", None)


def count_leaves(node):
    if node["is_leaf"]:
        return 1
    return count_leaves(node["left"]) + count_leaves(node["right"])


def tree_depth(node):
    if node["is_leaf"]:
        return 0
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))
