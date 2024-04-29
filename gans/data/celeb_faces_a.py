# Dataset class for the COCO dataset
# Mostly taken from here: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
import glob
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class CelebFacesA(Dataset):    
    """TODO.

    Dataset can be found here https://www.kaggle.com/datasets/jessicali9530/celeba-dataset.
    """

    def __init__(self, dataset_root: str, dataset_split: str = "train", transforms: T = None):
        """Initialize the CelebFacesA Dataset

        Args:
            dataset_root: Path to the dataset root
            dataset_split: which dataset split to use; `train`, `val`, `test`
        """
        self._transforms = transforms
        self._images = self._get_files(dataset_root, dataset_split)

    def __getitem__(self, index) -> Image:
        """Retrieve and preprocess samples from the dataset"""

        _image = Image.open(self._images[index]).convert("RGB")

        # Preprocess the input data before passing it to the model
        if self._transforms is not None:
            image, target = self._transforms(_image)

        return _image


def _make_celebfacesa_transforms(dataset_split: str) -> T:
    """Initialize transforms for the CelebFacesA dataset.

    Args:
        dataset_split: which dataset split to use; `train`, `val`, `test`

    """
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    img_size = (64, 64)

    if dataset_split == "train":
        return T.Compose(
            [
                T.Resize(img_size),
                T.CenterCrop(img_size), # this doesn't do anything if the img_size are the same
                normalize,
            ]
        )

    if dataset_split == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {dataset_split}")


def _get_file_paths(dataset_root: str, split: str):
    """TODO
    
    Args:
        dataset_root: Path to the dataset root dir
        split: `train`, `val`, or `test`
    """
    dataset_path = Path(dataset_root)

    if split == "train":
        images_root = dataset_path / "images"
    
    image_paths = glob.glob(f"{images_root}")
    return image_paths


def build_CelebFacesA(
    dataset_root: str,
    dataset_split: str = "train",
) -> Dataset:
    """Initializes the CelebFacesA dataset class

    Args:
        root: Full path to the dataset root
        split: which dataset split to use; `train`, `val`, or `test`
    """
    coco_root = Path(dataset_root)

    # Create the data augmentation transforms
    data_transforms = _make_celebfacesa_transforms(dataset_split)

    dataset = CelebFacesA(dataset_root=dataset_root, dataset_split=dataset_split, transforms=data_transforms)

    return dataset
