"""Dataset construction for the experiment harness.

Returns a small DatasetSpec carrying the loader-ready train/val datasets
plus image channel count and number of downsampling stages so the
encoder/decoder can be built to land on an 8x8 latent grid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass
class DatasetSpec:
    train: Dataset
    val: Dataset
    in_channels: int
    n_downsample: int
    image_size: int


def build_dataset(name: str, data_root: str) -> DatasetSpec:
    name = name.lower()
    data_root = os.path.expanduser(data_root)

    if name == 'cifar10':
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.CIFAR10(root = data_root, train = True, download = True, transform = tfm)
        val = datasets.CIFAR10(root = data_root, train = False, download = True, transform = tfm)
        return DatasetSpec(train, val, in_channels = 3, n_downsample = 2, image_size = 32)

    if name == 'cifar100':
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.CIFAR100(root = data_root, train = True, download = True, transform = tfm)
        val = datasets.CIFAR100(root = data_root, train = False, download = True, transform = tfm)
        return DatasetSpec(train, val, in_channels = 3, n_downsample = 2, image_size = 32)

    if name == 'celeba':
        tfm = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.CelebA(root = data_root, split = 'train', download = True, transform = tfm)
        val = datasets.CelebA(root = data_root, split = 'valid', download = True, transform = tfm)
        return DatasetSpec(train, val, in_channels = 3, n_downsample = 3, image_size = 64)

    if name == 'stl10':
        # 96x96 natural images; resize to 64x64 to match CelebA encoder depth.
        # Use unlabeled+train split (105k images) for training; test split (8k) for val.
        tfm = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.STL10(root = data_root, split = 'train+unlabeled', download = True, transform = tfm)
        val = datasets.STL10(root = data_root, split = 'test', download = True, transform = tfm)
        return DatasetSpec(train, val, in_channels = 3, n_downsample = 3, image_size = 64)

    if name == 'stl10_labeled':
        # labeled-only STL-10 (10 classes) for the classification probe. The VAE is
        # trained on 'stl10' (train+unlabeled); the probe needs labels, so use the
        # 5k labeled train split and the 8k test split.
        tfm = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.STL10(root = data_root, split = 'train', download = True, transform = tfm)
        val = datasets.STL10(root = data_root, split = 'test', download = True, transform = tfm)
        return DatasetSpec(train, val, in_channels = 3, n_downsample = 3, image_size = 64)

    if name == 'fashion_mnist':
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train = datasets.FashionMNIST(root = data_root, train = True, download = True, transform = tfm)
        val = datasets.FashionMNIST(root = data_root, train = False, download = True, transform = tfm)
        return DatasetSpec(train, val, in_channels = 1, n_downsample = 2, image_size = 28)

    if name == 'tiny_imagenet':
        return _build_tiny_imagenet(data_root)

    raise ValueError(f'unknown dataset: {name}')


def _build_tiny_imagenet(data_root: str) -> DatasetSpec:
    """Tiny ImageNet: 200 classes, 64x64, 100k train / 10k val.

    Downloads from cs231n.stanford.edu if not already present.
    Reorganises the flat val/images/ directory into per-class subdirectories
    on first use (required for ImageFolder).
    """
    import shutil
    import urllib.request
    import zipfile
    from pathlib import Path

    root = Path(data_root) / 'tiny-imagenet-200'
    zip_path = Path(data_root) / 'tiny-imagenet-200.zip'

    if not root.exists():
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        print(f'Downloading Tiny ImageNet from {url} ...')
        urllib.request.urlretrieve(url, zip_path)
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_root)
        zip_path.unlink(missing_ok=True)

    # Reorganise val: flat images/ + val_annotations.txt → per-class subdirs
    val_org = root / 'val_organized'
    if not val_org.exists():
        print('Reorganising Tiny ImageNet val split...')
        ann_path = root / 'val' / 'val_annotations.txt'
        img_dir  = root / 'val' / 'images'
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                fname, cls = parts[0], parts[1]
                dst = val_org / cls
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_dir / fname, dst / fname)

    tfm_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tfm_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.ImageFolder(str(root / 'train'), transform=tfm_train)
    val   = datasets.ImageFolder(str(val_org),        transform=tfm_val)
    return DatasetSpec(train, val, in_channels=3, n_downsample=3, image_size=64)
