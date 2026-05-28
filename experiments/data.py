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

    if name == 'fashion_mnist':
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train = datasets.FashionMNIST(root = data_root, train = True, download = True, transform = tfm)
        val = datasets.FashionMNIST(root = data_root, train = False, download = True, transform = tfm)
        return DatasetSpec(train, val, in_channels = 1, n_downsample = 2, image_size = 28)

    raise ValueError(f'unknown dataset: {name}')
