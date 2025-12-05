"""
JSON-based dataset loader for IQA datasets.
This module provides a flexible dataset class that loads image paths and scores from JSON files.
"""
import json
import os

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def pil_loader(path):
    """Load image using PIL."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class JSONImageDataset(data.Dataset):
    """
    Dataset class for loading images and scores from JSON files.
    
    JSON format expected:
    [
        {
            "image": "relative/path/to/image.jpg",
            "score": 75.5
        },
        ...
    ]
    
    Args:
        json_path: Path to the JSON file containing image paths and scores
        root_dir: Root directory for dataset images (will be prepended to image paths)
        transform: torchvision transforms to apply to images
        patch_num: Number of patches to sample from each image (for data augmentation)
    """
    
    def __init__(self, json_path, root_dir, transform, patch_num=1):
        self.root_dir = root_dir
        self.transform = transform
        self.patch_num = patch_num
        
        # Load JSON file
        with open(json_path, 'r') as f:
            data_list = json.load(f)
        
        # Create samples list with repeated entries for patch sampling
        self.samples = []
        for item in data_list:
            img_path = os.path.join(root_dir, item['image'])
            score = float(item['score'])
            # Repeat each image patch_num times for multiple patch sampling
            for _ in range(patch_num):
                self.samples.append((img_path, score))
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        
        Returns:
            tuple: (sample, target) where sample is the transformed image 
                   and target is the quality score.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target
    
    def __len__(self):
        return len(self.samples)


def get_transforms(patch_size=224, is_train=True):
    """
    Get standard transforms for IQA datasets.
    
    Args:
        patch_size: Size of the cropped patch
        is_train: Whether to apply training augmentations
    
    Returns:
        torchvision.transforms.Compose object
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((512, 384)),
            transforms.RandomCrop(size=patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.RandomCrop(size=patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225))
        ])
