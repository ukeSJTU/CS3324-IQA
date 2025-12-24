"""
Simplified dataset loader for JSON-based IQA datasets.
Supports: koniq_train, koniq_test, spaq_test, kadid_test, agiqa_test
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import json
import os


class IQADataset(data.Dataset):
    """Generic IQA Dataset loader for JSON metadata format"""

    def __init__(self, dataset_root, json_path, patch_size=224, patch_num=1,
                 is_train=True, resize_size=(512, 384)):
        """
        Args:
            dataset_root: Root directory containing image folders (e.g., /path/to/datasets/)
            json_path: Path to JSON metadata file
            patch_size: Size of cropped patches
            patch_num: Number of patches to sample per image
            is_train: Whether this is training data (affects augmentation)
            resize_size: Size to resize images before cropping (None to skip)
        """
        self.dataset_root = dataset_root
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.is_train = is_train

        # Load metadata
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Build transforms
        transform_list = []

        if is_train:
            transform_list.append(transforms.RandomHorizontalFlip())

        if resize_size is not None:
            transform_list.append(transforms.Resize(resize_size))

        if is_train:
            transform_list.append(transforms.RandomCrop(size=patch_size))
        else:
            transform_list.append(transforms.RandomCrop(size=patch_size))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225))
        ])

        self.transform = transforms.Compose(transform_list)

        # Expand dataset to include multiple patches per image
        self.samples = []
        for item in self.data:
            img_path = os.path.join(dataset_root, item['image'])
            score = float(item['score'])  # Handle both string and float scores
            for _ in range(patch_num):
                self.samples.append((img_path, score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, score = self.samples[idx]

        # Load image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # Apply transforms
        img = self.transform(img)

        return img, score


if __name__ == "__main__":
    """Quick test to verify dataset loading works"""

    # Test parameters
    dataset_root = '../datasets/'
    json_path = '../datasets/metas/koniq_train.json'

    print("Testing IQADataset loading...")
    print(f"Dataset root: {dataset_root}")
    print(f"JSON path: {json_path}")
    print()

    # Create dataset (small patch_num for quick test)
    dataset = IQADataset(
        dataset_root=dataset_root,
        json_path=json_path,
        patch_size=224,
        patch_num=2,  # Small for testing
        is_train=True,
        resize_size=(512, 384)
    )

    print(f"Dataset size: {len(dataset)} samples")
    print()

    # Test loading a few samples
    print("Loading first 3 samples...")
    for i in range(3):
        img, score = dataset[i]
        print(f"Sample {i}: img shape={img.shape}, score={score:.2f}")

    print()
    print("Testing DataLoader...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )

    batch = next(iter(dataloader))
    imgs, scores = batch
    print(f"Batch: imgs shape={imgs.shape}, scores shape={scores.shape}")
    print(f"Scores in batch: {scores.tolist()}")

    print("\n✓ Dataset loading works correctly!")
