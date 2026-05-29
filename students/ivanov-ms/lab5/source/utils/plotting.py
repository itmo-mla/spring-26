import os
import matplotlib.pyplot as plt

IMAGES_DIR = "./images/"


def _save_and_close(img_name: str):
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
        print(f"Created directory for images: {os.path.abspath(IMAGES_DIR)}")

    img_path = os.path.join(IMAGES_DIR, img_name)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close('all')


def plot_learning_curve_iterations(train_scores, name: str, img_name="learning_curve_iterations.png"):
    plt.figure(figsize=(10, 12))

    epochs = train_scores["epoch"]
    losses = [(train_scores["loss"], "Loss"), (train_scores["loss_reg"], "Loss with Reg.")]

    for i, (loss, loss_name) in enumerate(losses):
        plt.subplot(2, 1, i + 1)
        plt.plot(epochs, loss, '-', label=f"Training {loss_name} {name}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Learning Curve: {loss_name} {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

    _save_and_close(img_name)
