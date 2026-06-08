"""Dataset construction for the experiment harness.

Returns a small DatasetSpec carrying the loader-ready train/val datasets
plus image channel count and number of downsampling stages so the
encoder/decoder can be built to land on an 8x8 latent grid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
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

    if name == 'pcam':
        return _build_pcam(data_root)
    
    if name == 'galaxy10_decals':
        return _build_galaxy10_decals("experiments/space/galaxy.h5", data_root)

    if name == 'dtd':
        tfm = transforms.Compose([
            transforms.Resize(64), transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.DTD(root=data_root, split='train', partition=1, download=True, transform=tfm)
        val = datasets.DTD(root=data_root, split='val', partition=1, download=True, transform=tfm)

        return DatasetSpec(train, val, in_channels=3, n_downsample=3, image_size=64)

    if name == 'omniglot':
        tfm = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train = datasets.Omniglot(root=data_root, background=True, download=True, transform=tfm)
        val = datasets.Omniglot(root=data_root, background=False, download=True, transform=tfm)
        
        return DatasetSpec(train, val, in_channels=1, n_downsample=3, image_size=64)
        
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

def _build_galaxy10_decals(h5_path: str, data_root: str) -> DatasetSpec:
    """Galaxy10 DECals: manual extraction from dataset stored locally.
    """
    import os
    import h5py
    import torch
    from PIL import Image

    root_dir = os.path.join(data_root, "galaxy10_extracted")
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")

    if not os.path.exists(root_dir):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Galaxy10 file not found at {h5_path}.")

        print("Extracting Galaxy10 .h5 into class folders for fast loading...")
        with h5py.File(h5_path, "r") as f:
            images = f["images"][:]
            labels = f["ans"][:]

        total_images = len(labels)

        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_images, generator=g).tolist()
        split_idx = int(total_images * 0.9)

        train_indices = set(indices[:split_idx])

        for idx in range(total_images):
            is_train = idx in train_indices
            target_dir = train_dir if is_train else val_dir

            class_dir = os.path.join(target_dir, f"class_{int(labels[idx])}")
            os.makedirs(class_dir, exist_ok=True)

            img = Image.fromarray(images[idx])
            img.save(os.path.join(class_dir, f"img_{idx}.png"))

        print("Extraction complete!")

    tfm_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tfm_val = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train = datasets.ImageFolder(train_dir, transform=tfm_train)
    val = datasets.ImageFolder(val_dir, transform=tfm_val)

    return DatasetSpec(train, val, in_channels=3, n_downsample=3, image_size=64)

def _build_pcam(data_root) -> DatasetSpec:
    """PatchCamelyon (PCAM): manual extraction from dataset stored locally.
    """
    import os
    import zipfile
    import h5py
    from torch.utils.data import Dataset
    from PIL import Image

    root_dir = os.path.join(data_root, "pcam_extracted")
    zip_path = os.path.join("experiments", "medical", "pcam.zip")

    internal_files = {
        "train_x": "pcam/training_split.h5",
        "train_y": "Labels/Labels/camelyonpatch_level_2_split_train_y.h5",
        "val_x": "pcam/validation_split.h5",
        "val_y": "Labels/Labels/camelyonpatch_level_2_split_valid_y.h5",
    }

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Base archive not found at {zip_path}")

    os.makedirs(root_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for key, internal_path in internal_files.items():
            dest_path = os.path.join(root_dir, f"{key}.h5")
            expected_size = z.getinfo(internal_path).file_size
            
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) != expected_size:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                
                print(f"Extracting {key}.h5 from zip archive...")
                with z.open(internal_path) as source, open(dest_path, "wb") as target:
                    import shutil
                    shutil.copyfileobj(source, target)
                    
    print("HDF5 extraction validation complete!")
    class H5PCAMDataset(Dataset): 
        def __init__(self, h5_x_path, h5_y_path, transform=None):
            self.h5_x_path = h5_x_path
            self.h5_y_path = h5_y_path
            self.transform = transform
            
            with h5py.File(self.h5_y_path, "r") as fy:
                self.y_key = list(fy.keys())[0]
                self.length = len(fy[self.y_key])
                
            self.fx = None
            self.fy = None

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            if self.fx is None or self.fy is None:
                self.fx = h5py.File(self.h5_x_path, "r")
                self.fy = h5py.File(self.h5_y_path, "r")
                self.x_key = list(self.fx.keys())[0]
                self.y_key = list(self.fy.keys())[0]

            img_array = self.fx[self.x_key][idx]
            lbl = self.fy[self.y_key][idx]
            label_val = int(lbl.item() if hasattr(lbl, 'item') else lbl.ravel()[0])

            img = Image.fromarray(img_array)
            if self.transform:
                img = self.transform(img)

            return img, label_val

    tfm_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train = H5PCAMDataset(
        os.path.join(root_dir, "train_x.h5"), 
        os.path.join(root_dir, "train_y.h5"), 
        transform=tfm_train
    )
    val = H5PCAMDataset(
        os.path.join(root_dir, "val_x.h5"), 
        os.path.join(root_dir, "val_y.h5"), 
        transform=tfm_val
    )

    return DatasetSpec(train, val, in_channels=3, n_downsample=3, image_size=64)