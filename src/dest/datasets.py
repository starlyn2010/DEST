"""
datasets.py — Dataset loading with proper augmentation and TinyImageNet support.

Augmentation policy (train only; test uses only normalization):
  MNIST / FashionMNIST : none (28x28, grayscale, easy enough)
  CIFAR-10  : RandomCrop(32, pad=4) + RandomHorizontalFlip
  CIFAR-100 : RandomCrop(32, pad=4) + RandomHorizontalFlip
  TinyImageNet : RandomCrop(64, pad=8) + RandomHorizontalFlip

All augmentations are applied identically to every sampler — the only
variable between experiments is the ORDER in which samples are presented.
"""

import os
import shutil
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# ───────────────────────── normalization constants ──────────────────────────
_STATS = {
    "MNIST":        {"mean": (0.1307,),                  "std": (0.3081,)},
    "FASHIONMNIST": {"mean": (0.2860,),                  "std": (0.3530,)},
    "CIFAR10":      {"mean": (0.4914, 0.4822, 0.4465),   "std": (0.2023, 0.1994, 0.2010)},
    "CIFAR100":     {"mean": (0.5071, 0.4867, 0.4408),   "std": (0.2675, 0.2565, 0.2761)},
    "TINYIMAGENET": {"mean": (0.4802, 0.4481, 0.3975),   "std": (0.2302, 0.2265, 0.2262)},
}


def _build_transforms(dataset_name: str, input_size: int):
    """Return (train_transform, test_transform) for the given dataset."""
    name = dataset_name.upper()
    mean = _STATS[name]["mean"]
    std  = _STATS[name]["std"]

    normalize = transforms.Normalize(mean, std)

    if name in ("MNIST", "FASHIONMNIST"):
        base = [transforms.ToTensor(), normalize]
        return transforms.Compose(base), transforms.Compose(base)

    pad = input_size // 8  # 4 for 32-px, 8 for 64-px
    train_tf = transforms.Compose([
        transforms.RandomCrop(input_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, test_tf


# ─────────────────────────── TinyImageNet helpers ───────────────────────────
_TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _organize_tinyimagenet_val(data_dir: str) -> None:
    """Move TinyImageNet val images into per-class subdirectories."""
    val_dir = os.path.join(data_dir, "val")
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.exists(ann_file):
        return
    # Check if already organized (skip if class dirs exist)
    if any(
        os.path.isdir(os.path.join(val_dir, d))
        for d in os.listdir(val_dir)
        if d.startswith("n")
    ):
        return
    print("Organizing TinyImageNet val set...")
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name, cls = parts[0], parts[1]
            cls_dir = os.path.join(val_dir, cls, "images")
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(val_dir, "images", img_name)
            dst = os.path.join(cls_dir, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)


def download_tinyimagenet(root: str = "./data") -> str | None:
    """Download and extract TinyImageNet. Returns path or None on failure."""
    extract_path = os.path.join(root, "tiny-imagenet-200")
    if os.path.exists(extract_path):
        _organize_tinyimagenet_val(extract_path)
        return extract_path
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    try:
        print(f"Downloading TinyImageNet (~237 MB)…")
        urllib.request.urlretrieve(_TINYIMAGENET_URL, zip_path)
        print("Extracting…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        os.remove(zip_path)
        _organize_tinyimagenet_val(extract_path)
        return extract_path
    except Exception as exc:
        print(f"⚠️  TinyImageNet download failed ({exc}). Skipping dataset.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None


# ─────────────────────────── main loader class ──────────────────────────────
class DatasetLoader:
    """
    Unified dataset loader.  Returns stratified train / val / test splits with
    correct per-dataset augmentation and normalization.

    Parameters
    ----------
    dataset_name : str
        One of MNIST, FASHIONMNIST, CIFAR10, CIFAR100, TINYIMAGENET.
    val_fraction : float
        Fraction of training data reserved for validation.
    split_seed : int
        Seed for the deterministic stratified split (same across all samplers).
    data_root : str
        Directory where datasets are downloaded / cached.
    """

    def __init__(
        self,
        dataset_name: str = "MNIST",
        val_fraction: float = 0.1,
        split_seed: int = 0,
        data_root: str = "./data",
    ):
        self.dataset_name = dataset_name.upper()
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.data_root = data_root

    # ------------------------------------------------------------------
    def get_datasets(self):
        """
        Returns
        -------
        train_dataset, val_dataset, test_dataset, n_classes, input_shape, available
        available : bool  — False only for TinyImageNet when download fails.
        """
        name = self.dataset_name

        if name == "TINYIMAGENET":
            return self._load_tinyimagenet()

        # ── standard torchvision datasets ──────────────────────────────
        input_size = 28 if name in ("MNIST", "FASHIONMNIST") else 32
        train_tf, test_tf = _build_transforms(name, input_size)

        kw = {"root": self.data_root, "download": True}
        if name == "MNIST":
            train_full   = datasets.MNIST(train=True,  transform=train_tf, **kw)
            test_dataset = datasets.MNIST(train=False, transform=test_tf,  **kw)
            n_classes, input_shape = 10, (1, 28, 28)

        elif name == "FASHIONMNIST":
            train_full   = datasets.FashionMNIST(train=True,  transform=train_tf, **kw)
            test_dataset = datasets.FashionMNIST(train=False, transform=test_tf,  **kw)
            n_classes, input_shape = 10, (1, 28, 28)

        elif name == "CIFAR10":
            train_full   = datasets.CIFAR10(train=True,  transform=train_tf, **kw)
            test_dataset = datasets.CIFAR10(train=False, transform=test_tf,  **kw)
            n_classes, input_shape = 10, (3, 32, 32)

        elif name == "CIFAR100":
            train_full   = datasets.CIFAR100(train=True,  transform=train_tf, **kw)
            test_dataset = datasets.CIFAR100(train=False, transform=test_tf,  **kw)
            n_classes, input_shape = 100, (3, 32, 32)

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        train_ds, val_ds = self._stratified_split(train_full)
        return train_ds, val_ds, test_dataset, n_classes, input_shape, True

    # ------------------------------------------------------------------
    def _load_tinyimagenet(self):
        data_dir = download_tinyimagenet(self.data_root)
        if data_dir is None:
            return None, None, None, None, None, False  # auto-skip

        train_tf, test_tf = _build_transforms("TINYIMAGENET", 64)
        from torchvision.datasets import ImageFolder

        train_root = os.path.join(data_dir, "train")
        val_root   = os.path.join(data_dir, "val")

        train_full   = ImageFolder(train_root, transform=train_tf)
        test_dataset = ImageFolder(val_root,   transform=test_tf)

        n_classes   = len(train_full.classes)
        input_shape = (3, 64, 64)

        train_ds, val_ds = self._stratified_split(train_full)
        return train_ds, val_ds, test_dataset, n_classes, input_shape, True

    # ------------------------------------------------------------------
    def _stratified_split(self, train_full):
        targets = getattr(train_full, "targets", None)
        if targets is None:
            # ImageFolder stores .targets as a list
            targets = [s[1] for s in train_full.samples]
        if torch.is_tensor(targets):
            targets = targets.numpy()
        targets = np.array(targets)

        train_idx, val_idx = train_test_split(
            np.arange(len(train_full)),
            test_size=self.val_fraction,
            random_state=self.split_seed,
            stratify=targets,
        )
        return Subset(train_full, train_idx), Subset(train_full, val_idx)
