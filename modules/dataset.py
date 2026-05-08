"""
StegoDetect - PyTorch Dataset & DataLoader for Module 1 CNN training.

Provides:
    - StegoDataset: Custom Dataset for 3-class stego classification
    - get_transforms: Augmentation & normalization pipelines
    - get_dataloaders: Ready-to-use train/val/test DataLoaders
"""

import os
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Get image transforms for the given split.

    NOTE: We do NOT use ColorJitter, RandomRotation, RandomCrop, or any
    transform that modifies pixel values or spatial arrangement —
    these destroy steganographic signatures.
    """
    cfg = config.CNN_CONFIG
    mean, std = cfg["mean"], cfg["std"]
    size = cfg["input_size"]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  # val, test
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ──────────────────────────────────────────────────────────────────────
# StegoDataset
# ──────────────────────────────────────────────────────────────────────

class StegoDataset(Dataset):
    """
    PyTorch Dataset for 3-class steganography classification.

    Folder structure expected:
        root_dir/
            {split}/
                clean/   → label 0
                lsb/     → label 1
                pvd/     → label 2
    """

    def __init__(
        self,
        root_dir: str = None,
        split: str = "train",
        transform=None,
        use_noise_residual: bool = False,
    ):
        """
        Args:
            root_dir: Path to the combined dataset (default: config.COMBINED_DIR)
            split: 'train', 'val', or 'test'
            transform: torchvision transforms to apply
            use_noise_residual: If True, add a 4th Laplacian noise-residual channel
        """
        self.root_dir = Path(root_dir) if root_dir else config.COMBINED_DIR
        self.split = split
        self.transform = transform or get_transforms(split)
        self.use_noise_residual = use_noise_residual

        self.class_names = config.CLASS_NAMES  # ["clean", "lsb", "pvd"]
        self.class_map = config.CLASS_MAP      # {"clean": 0, "lsb": 1, "pvd": 2}

        # Build sample list: (image_path, label)
        self.samples = []
        split_dir = self.root_dir / split

        for class_name, label in self.class_map.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"  Warning: {class_dir} does not exist, skipping.")
                continue
            for img_file in sorted(class_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self.samples.append((img_file, label))

        # Compute class weights for weighted loss: total / (num_classes * count)
        label_counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        num_classes = len(self.class_names)
        self.class_weights = torch.tensor([
            total / (num_classes * label_counts.get(i, 1))
            for i in range(num_classes)
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    # SRM (Spatial Rich Model) high-pass kernels for steganalysis
    # These are standard filters used in academic steganalysis research
    _SRM_KERNEL_1 = np.array([[ 0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0],
                               [ 0,  1, -2,  1,  0],
                               [ 0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0]], dtype=np.float64)

    _SRM_KERNEL_2 = np.array([[ 0,  0,  0,  0,  0],
                               [ 0, -1,  2, -1,  0],
                               [ 0,  2, -4,  2,  0],
                               [ 0, -1,  2, -1,  0],
                               [ 0,  0,  0,  0,  0]], dtype=np.float64)

    _SRM_KERNEL_3 = np.array([[-1,  2, -2,  2, -1],
                               [ 2, -6,  8, -6,  2],
                               [-2,  8, -12, 8, -2],
                               [ 2, -6,  8, -6,  2],
                               [-1,  2, -2,  2, -1]], dtype=np.float64)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms (resize, flip, normalize for RGB)
        image_tensor = self.transform(image)

        # Optionally add noise-residual as 4th channel
        if self.use_noise_residual:
            # Resize image FIRST so residual is spatially aligned with RGB tensor
            size = config.CNN_CONFIG["input_size"]
            img_resized = image.resize((size, size), Image.BILINEAR)
            img_np = np.array(img_resized, dtype=np.float64)

            # Compute SRM residuals on all 3 channels, then average
            # This captures per-channel bit-level artifacts from LSB embedding
            residuals = []
            for kernel in [self._SRM_KERNEL_1, self._SRM_KERNEL_2, self._SRM_KERNEL_3]:
                channel_residuals = []
                for c in range(3):
                    filtered = cv2.filter2D(img_np[:, :, c], cv2.CV_64F, kernel)
                    channel_residuals.append(filtered)
                # Average across RGB channels
                residuals.append(np.mean(channel_residuals, axis=0))

            # Average the 3 SRM kernel responses
            noise_map = np.mean(residuals, axis=0)

            # Standardize with fixed statistics (zero-mean, unit-variance)
            # Using fixed scale preserves magnitude differences between clean/stego
            # Typical SRM residual std is ~2-5 for natural images
            noise_map = noise_map / 4.0  # Scale to roughly [-1, 1] range

            # Clip extreme outliers
            noise_map = np.clip(noise_map, -3.0, 3.0)

            # Convert to tensor and concatenate as 4th channel
            noise_channel = torch.tensor(noise_map, dtype=torch.float32).unsqueeze(0)
            image_tensor = torch.cat([image_tensor, noise_channel], dim=0)

        return image_tensor, label

    def get_class_distribution(self) -> dict:
        """Return the count of samples per class."""
        counts = Counter(label for _, label in self.samples)
        return {self.class_names[i]: counts.get(i, 0) for i in range(len(self.class_names))}


# ──────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────

def get_dataloaders(
    root_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    use_noise_residual: bool = False,
):
    """
    Create train, val, and test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    cfg = config.CNN_CONFIG
    bs = batch_size or cfg["batch_size"]
    nw = num_workers or cfg["num_workers"]

    train_dataset = StegoDataset(root_dir, split="train", use_noise_residual=use_noise_residual)
    val_dataset = StegoDataset(root_dir, split="val", use_noise_residual=use_noise_residual)
    test_dataset = StegoDataset(root_dir, split="test", use_noise_residual=use_noise_residual)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.class_weights


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("StegoDetect - Dataset Statistics")
    print("=" * 50)

    for split in ["train", "val", "test"]:
        ds = StegoDataset(split=split)
        dist = ds.get_class_distribution()
        total = len(ds)
        print(f"\n{split.upper()} split: {total} samples")
        for cls, count in dist.items():
            pct = 100 * count / total if total > 0 else 0
            print(f"  {cls:>6}: {count:>6} ({pct:.1f}%)")
        print(f"  Class weights: {ds.class_weights.tolist()}")

    # Quick test: load one batch
    print("\nLoading one training batch...")
    train_loader, _, _, _ = get_dataloaders(num_workers=0)
    images, labels = next(iter(train_loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels: {labels.tolist()[:10]}...")
    print("\n[OK] Dataset module working correctly!")
