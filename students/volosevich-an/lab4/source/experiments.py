# %%
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from data_loader import load_data
from gmm import EMGMM
from metrics import clustering_accuracy, bic_score, aic_score, compute_log_likelihood

# %%
X_train, X_test, y_train, y_test, feature_names = load_data(
    csv_path="../data/breast-cancer.csv",
    test_size=0.2,
    random_state=22)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%
n_components = 2
n_samples, n_features = X_train.shape

# %%
start_time = time.time()

custom_gmm = EMGMM(
    n_components=n_components,
    max_iter=100,
    tol=1e-4,
    random_state=22
)

custom_gmm.fit(X_train)
custom_time = time.time() - start_time
custom_time

# %%
custom_train_preds = custom_gmm.predict(X_train)
custom_test_preds = custom_gmm.predict(X_test)

custom_ll = custom_gmm.log_likelihood_history_[-1]

n_params = (
    (n_components - 1)
    + n_components * n_features
    + n_components * (n_features * (n_features + 1) / 2)
)

custom_bic = bic_score(custom_ll, n_params, X_train.shape[0])
custom_aic = aic_score(custom_ll, n_params)

custom_acc = clustering_accuracy(y_test, custom_test_preds)
custom_acc, custom_ll, custom_bic, custom_aic

# %%
start_time = time.time()

sklearn_gmm = GaussianMixture(
    n_components=n_components,
    covariance_type='full',
    max_iter=100,
    random_state=22
)

sklearn_gmm.fit(X_train)
sklearn_time = time.time() - start_time

sklearn_time


# %%
sklearn_train_preds = sklearn_gmm.predict(X_train)
sklearn_test_preds = sklearn_gmm.predict(X_test)

sklearn_ll = sklearn_gmm.score(X_train) * len(X_train)

sklearn_bic = sklearn_gmm.bic(X_train)
sklearn_aic = sklearn_gmm.aic(X_train)

sklearn_acc = clustering_accuracy(y_test, sklearn_test_preds)

sklearn_acc, sklearn_ll, sklearn_bic, sklearn_aic

# %%
print("Custom GMM:")
print(f"Log-Likelihood (train): {custom_ll:.4f}")
print(f"BIC: {custom_bic:.4f}")
print(f"AIC: {custom_aic:.4f}")
print(f"Test Accuracy: {custom_acc:.4f}")
print(f"Runtime: {custom_time:.4f} sec\n")

print("Sklearn GMM:")
print(f"Log-Likelihood (train): {sklearn_ll:.4f}")
print(f"BIC: {sklearn_bic:.4f}")
print(f"AIC: {sklearn_aic:.4f}")
print(f"Test Accuracy: {sklearn_acc:.4f}")
print(f"Runtime: {sklearn_time:.4f} sec")

# %%
plt.figure(figsize=(10, 6))
plt.plot(custom_gmm.log_likelihood_history_)
plt.title("EM Convergence (Train Log-Likelihood)")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.grid(True)
plt.show()

# %%
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

x_comp = 0
y_comp = 1

plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[:, x_comp], X_train_pca[:, y_comp], c=custom_train_preds)
plt.title("Custom GMM Clusters (Train PCA)")
plt.xlabel(f"PC{x_comp + 1}")
plt.ylabel(f"PC{y_comp + 1}")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[:, x_comp], X_train_pca[:, y_comp], c=sklearn_train_preds)
plt.title("Sklearn GMM Clusters (Train PCA)")
plt.xlabel(f"PC{x_comp + 1}")
plt.ylabel(f"PC{y_comp + 1}")
plt.show()


