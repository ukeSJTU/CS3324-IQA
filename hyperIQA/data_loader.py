"""
Data loader module for IQA datasets.

This module provides a unified interface for loading different IQA datasets
with appropriate preprocessing transforms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T

import folders

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Dataset categories by preprocessing requirements
STANDARD_DATASETS = {"live", "csiq", "tid2013", "livec"}
KONIQ_RESIZE = (512, 384)
BID_RESIZE = (512, 512)

# Dataset name to folder class mapping
DATASET_FOLDER_MAPPING = {
    "live": folders.LIVEFolder,
    "livec": folders.LIVEChallengeFolder,
    "csiq": folders.CSIQFolder,
    "koniq-10k": folders.Koniq10kFolder,
    "bid": folders.BIDFolder,
    "tid2013": folders.TID2013Folder,
}


def _create_transforms(
    patch_size: int,
    is_train: bool,
    resize: tuple[int, int] | None = None,
) -> T.Compose:
    """
    Create image transforms for training or testing.

    Args:
        patch_size: Size of random crop.
        is_train: Whether to include training augmentations.
        resize: Optional resize dimensions (height, width).

    Returns:
        Composed transform pipeline.
    """
    transforms_list: list[T.Transform] = []

    if is_train:
        transforms_list.append(T.RandomHorizontalFlip())

    if resize is not None:
        transforms_list.append(T.Resize(resize))

    transforms_list.extend([
        T.RandomCrop(size=patch_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return T.Compose(transforms_list)


class IQADataLoader:
    """
    Data loader class for IQA databases.

    Handles dataset instantiation and creates PyTorch DataLoader
    with appropriate transforms for different IQA benchmarks.
    """

    def __init__(
        self,
        dataset: str,
        path: str,
        img_indx: list[int],
        patch_size: int,
        patch_num: int,
        batch_size: int = 1,
        is_train: bool = True,
    ) -> None:
        """
        Initialize the data loader.

        Args:
            dataset: Dataset name ('live', 'csiq', 'tid2013', 'livec',
                     'koniq-10k', 'bid').
            path: Root path to the dataset.
            img_indx: List of image indices to include.
            patch_size: Size of random crop patches.
            patch_num: Number of patches per image.
            batch_size: Batch size for training.
            is_train: Whether this is for training (affects augmentation).

        Raises:
            ValueError: If dataset name is not recognized.
        """
        self.batch_size = batch_size
        self.is_train = is_train

        # Determine resize dimensions based on dataset
        resize = self._get_resize_dims(dataset)
        transforms = _create_transforms(patch_size, is_train, resize)

        # Get the appropriate folder class
        folder_class = DATASET_FOLDER_MAPPING.get(dataset)
        if folder_class is None:
            raise ValueError(
                f"Unknown dataset: {dataset}. "
                f"Supported: {list(DATASET_FOLDER_MAPPING.keys())}"
            )

        self.data = folder_class(
            root=path,
            index=img_indx,
            transform=transforms,
            patch_num=patch_num,
        )

    @staticmethod
    def _get_resize_dims(dataset: str) -> tuple[int, int] | None:
        """
        Get resize dimensions for a dataset.

        Args:
            dataset: Dataset name.

        Returns:
            Resize dimensions or None for standard datasets.
        """
        if dataset in STANDARD_DATASETS:
            return None
        elif dataset == "koniq-10k":
            return KONIQ_RESIZE
        elif dataset == "bid":
            return BID_RESIZE
        return None

    def get_data(self) -> TorchDataLoader:
        """
        Get a PyTorch DataLoader for the dataset.

        Returns:
            DataLoader with appropriate settings for train/test.
        """
        if self.is_train:
            return torch.utils.data.DataLoader(
                self.data,
                batch_size=self.batch_size,
                shuffle=True,
            )
        else:
            return torch.utils.data.DataLoader(
                self.data,
                batch_size=1,
                shuffle=False,
            )


# Legacy alias for backward compatibility
DataLoader = IQADataLoader
