import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

IMAGES_DIR = "./images/"


def _save_and_close(img_name: str):
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
        print(f"Created directory for images: {os.path.abspath(IMAGES_DIR)}")

    img_path = os.path.join(IMAGES_DIR, img_name)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close('all')


def plot_pca(model, X_train, X_test, y_train, y_test, img_name="gmm_pca.png"):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train.to_numpy())
    X_test = pca.transform(X_test.to_numpy())

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot data
    classes = np.unique(y_train)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, cls in enumerate(classes):
        cur_cls_train, cur_cls_test = y_train == cls, y_test == cls
        ax.scatter(
            X_train_pca[cur_cls_train, 0], X_train_pca[cur_cls_train, 1],
            c=colors[i], alpha=0.2, label=f"{cls} train"
        )
        ax.scatter(
            X_test[cur_cls_test, 0], X_test[cur_cls_test, 1],
            c=colors[i], alpha=1, marker="x", label=f"{cls} test"
        )

    # Plot gm circles
    mean_pca = pca.transform(model.means_)
    P = pca.components_.T  # Transformation matrix
    covs = model.covariances_
    covs_pca = []
    for i in range(len(covs)):
        transformed_cov = P.T @ covs[i] @ P
        covs_pca.append(transformed_cov)

    # Plot elipses
    colors = ["navy", "turquoise", "darkorange"] + colors
    for i in range(len(mean_pca)):
        v, w = np.linalg.eigh(covs_pca[i])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = Ellipse(
            mean_pca[i], v[0], v[1], angle=180 + angle, color=colors[i]
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")

    plt.legend()
    _save_and_close(img_name)


def plot_learning_curve_iterations(train_scores, name: str, img_name="learning_curve_iterations.png"):
    """
    Plot learning curve showing accuracy vs boosting iteration.
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(train_scores) + 1)
    plt.plot(iterations, train_scores, '-', label=f"Training {name}")
    plt.xlabel("Iteration")
    plt.ylabel(name)
    plt.title(f"Learning Curve: Training {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_and_close(img_name)
