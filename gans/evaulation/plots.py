import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_losses(
    gen_losses: list[float], disc_losses: list[float], save_path: str, dpi: int = 500
):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_losses, label="G")
    plt.plot(disc_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path, dpi=dpi)


def visualize_fixed_images(images_array: list[np.ndarray], save_path):
    # (B, C, H, W) -> (B, H, W, C)
    images_transposed = np.transpose(images_array, 2, 3, 1)
    fig, axes = plt.subplots(8, 8)
    for index, ax in enumerate(axes.flat):
        ax.imshow(images_transposed[index])
        ax.axis("off")

    fig.subplots_adjust(wspace=-0.70, hspace=0.05)
    fig.savefig(save_path, bbox_inches="tight")
