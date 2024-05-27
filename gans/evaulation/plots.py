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
    plt.close()


def visualize_fixed_images(images_array: np.ndarray, save_path):
    """TODO"""
    normalized_images = _normalize_batch_images(images_array)
    
    # (B, C, H, W) -> (B, H, W, C)
    images_transposed = np.transpose(normalized_images, axes=(0, 2, 3, 1))

    fig, axes = plt.subplots(8, 8)
    for index, ax in enumerate(axes.flat):
        ax.imshow(images_transposed[index])
        ax.axis("off")

    fig.subplots_adjust(wspace=-0.70, hspace=0.05)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def _normalize_batch_images(batched_images: np.ndarray):
    """Normalize images between [0,1]
    
    If using this for the DCGAN, the generated images will have range [-1, 1] and they
    will need to be normalized for visualization. There's probably a library to do this
    but doing it by scratch was more fun.

    Args:
        batched_images: a batch of numpy images which must have shape (B, 3, H, W)
    """
    # (B, 3, 1, 1)
    rgb_mins = batched_images.min(axis=(2,3))[:, :, np.newaxis, np.newaxis]
    rgb_maxes = batched_images.max(axis=(2,3))[:, :, np.newaxis, np.newaxis]

    normalized_images = (batched_images - rgb_mins) / (rgb_maxes - rgb_mins)
    return normalized_images

