import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


from GMM import GMM


def best_cluster_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)

    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    row_ind, col_ind = linear_sum_assignment(-matrix)

    mapping = {}
    for row, col in zip(row_ind, col_ind):
        mapping[labels_pred[col]] = labels_true[row]

    y_pred_mapped = np.array([mapping.get(label, label) for label in y_pred])

    return accuracy_score(y_true, y_pred_mapped), y_pred_mapped


iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

my_gmm = GMM(
    n_components=3,
    max_iter=100,
    tol=1e-4,
    random_state=42
)

my_gmm.fit(X_train_scaled)

my_train_ll = my_gmm.log_likelihood(X_train_scaled)
my_test_ll = my_gmm.log_likelihood(X_test_scaled)

my_train_score = my_gmm.score(X_train_scaled)
my_test_score = my_gmm.score(X_test_scaled)

my_pred = my_gmm.predict(X_test_scaled)
my_acc, my_pred_mapped = best_cluster_accuracy(y_test, my_pred)

print("My GMM")
print("Train log-likelihood:", my_train_ll)
print("Test log-likelihood:", my_test_ll)
print("Train mean log-likelihood:", my_train_score)
print("Test mean log-likelihood:", my_test_score)
print("Accuracy:", my_acc)

sklearn_gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    max_iter=100,
    tol=1e-4,
    random_state=42
)

sklearn_gmm.fit(X_train_scaled)

sk_train_ll = sklearn_gmm.score(X_train_scaled) * len(X_train_scaled)
sk_test_ll = sklearn_gmm.score(X_test_scaled) * len(X_test_scaled)

sk_train_score = sklearn_gmm.score(X_train_scaled)
sk_test_score = sklearn_gmm.score(X_test_scaled)

sk_pred = sklearn_gmm.predict(X_test_scaled)
sk_acc, sk_pred_mapped = best_cluster_accuracy(y_test, sk_pred)

print("Sklearn GMM")
print("Train log-likelihood:", sk_train_ll)
print("Test log-likelihood:", sk_test_ll)
print("Train mean log-likelihood:", sk_train_score)
print("Test mean log-likelihood:", sk_test_score)
print("Accuracy:", sk_acc)

plt.figure(figsize=(8, 5))
plt.plot(my_gmm.log_likelihood_history_)
plt.xlabel("Итерация")
plt.ylabel("Log-likelihood")
plt.title("Сходимость EM-алгоритма для GMM")
plt.grid(True)
plt.show()