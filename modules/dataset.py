"""
StegoDetect - Dataset & DataLoader  (v2 — fixed augmentation)

CRITICAL FIX:
  v1/v2 bug: Using color jitter / random brightness-contrast on stego images
  DESTROYS the LSB signal before training.  LSB embeds in the least-significant
  bit — a ±1 pixel value change.  Color jitter can shift pixel values by ±20+,
  completely erasing those bits.

  SAFE augmentations for stego detection:
    ✅  RandomHorizontalFlip  — flips spatial layout, pixel VALUES unchanged
    ✅  RandomVerticalFlip    — same
    ✅  RandomRotation(±5°)   — very gentle, minimal interpolation artefact
    ✅  Resize + CenterCrop   — avoids random resampling artefacts
    ❌  ColorJitter            — DESTROYS LSB signal
    ❌  RandomBrightnessContrast — DESTROYS LSB signal
    ❌  GaussianBlur           — blurs the noise residual
    ❌  JPEG compression       — overwrites LSB bits entirely
    ❌  RandomErasing          — erases regions, changes statistics

Expected directory layout:
    data/combined/
        train/
            clean/  lsb/  pvd/
        val/
            clean/  lsb/  pvd/
        test/
            clean/  lsb/  pvd/
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────────────
# Laplacian noise-residual helper
# ──────────────────────────────────────────────────────────────────────

def _compute_noise_residual(img_rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute a single-channel Laplacian residual from a (3, H, W) float tensor.

    The Laplacian acts as a high-pass filter that amplifies the subtle
    pixel-level noise introduced by LSB steganography.

    Steps:
      1. Convert to single-channel (mean of RGB channels)
      2. Apply discrete Laplacian via OpenCV
      3. Normalise to [-1, 1] range to match the backbone input scale

    Returns: (1, H, W) float tensor
    """
    # Convert to HxW uint8 for OpenCV
    # img_rgb_tensor is in [0,1] range (not yet normalised)
    gray = img_rgb_tensor.mean(dim=0).numpy()  # (H, W)
    gray_u8 = (gray * 255).clip(0, 255).astype(np.uint8)

    laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
    # Normalise to roughly [-1, 1]
    max_abs = np.abs(laplacian).max()
    if max_abs > 0:
        laplacian = laplacian / max_abs
    else:
        laplacian = np.zeros_like(laplacian)

    return torch.tensor(laplacian, dtype=torch.float32).unsqueeze(0)  # (1, H, W)


# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────

def _build_transforms(split: str, input_size: int = 224):
    """
    Build torchvision transform pipeline.

    Training:  gentle geometric-only augmentation (NO colour changes)
    Val/Test:  deterministic resize + centre crop only

    We deliberately skip ColorJitter, RandomBrightnessContrast, and
    GaussianBlur because they modify pixel VALUES and erase the
    LSB steganographic signal.
    """
    mean = config.CNN_CONFIG["mean"]
    std  = config.CNN_CONFIG["std"]

    normalise = transforms.Normalize(mean=mean, std=std)

    if split == "train":
        return transforms.Compose([
            # Resize slightly larger then crop — avoids hard edges from direct resize
            transforms.Resize(int(input_size * 1.05)),
            transforms.CenterCrop(input_size),   # deterministic crop (no random)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # Very gentle rotation — 5° max to limit interpolation artefacts
            transforms.RandomRotation(
                degrees=5,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ToTensor(),   # converts PIL [0,255] → float [0,1]
            normalise,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.05)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalise,
        ])


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class StegoDataset(Dataset):
    """
    Image dataset for LSB / PVD / clean steganalysis.

    Args:
        root_dir:           Path to split directory (e.g. data/combined/train)
        transform:          torchvision transform to apply BEFORE noise residual
        use_noise_residual: If True, append a Laplacian channel → 4-channel input
        class_map:          Dict mapping folder name → class index

    NOTE: The noise residual is computed on the pre-normalised float image
    (i.e. after ToTensor but before Normalize) to avoid scale issues.
    This is why we split the transform into base + normalise steps below.
    """

    def __init__(
        self,
        root_dir: Path,
        transform=None,
        use_noise_residual: bool = True,
        class_map: dict = None,
    ):
        self.root_dir           = Path(root_dir)
        self.use_noise_residual = use_noise_residual
        self.class_map          = class_map or config.CLASS_MAP
        self.transform          = transform

        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Scan directory structure and build (path, label) list."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        for class_name, label in self.class_map.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"  [WARN] Directory not found: {class_dir}")
                continue
            found = [
                p for p in class_dir.rglob("*")
                if p.suffix.lower() in valid_exts
            ]
            self.samples.extend([(p, label) for p in found])
            print(f"  {class_name}: {len(found)} images")
        print(f"  Total: {len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load image as RGB PIL
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Could not open {img_path}: {e}")
            # Return a black image with the correct label
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform is not None:
            img_tensor = self.transform(img)  # (3, H, W) normalised
        else:
            img_tensor = transforms.ToTensor()(img)

        if self.use_noise_residual:
            # Compute noise residual on the raw (un-normalised) image
            # We undo normalisation to get back to [0,1] range
            mean = torch.tensor(config.CNN_CONFIG["mean"]).view(3, 1, 1)
            std  = torch.tensor(config.CNN_CONFIG["std"]).view(3, 1, 1)
            raw  = img_tensor * std + mean   # approximately [0, 1]
            raw  = raw.clamp(0.0, 1.0)
            noise_ch = _compute_noise_residual(raw)
            img_tensor = torch.cat([img_tensor, noise_ch], dim=0)  # (4, H, W)

        return img_tensor, label

    def get_class_counts(self) -> dict:
        """Return {class_name: count} for this split."""
        label_counts = Counter(label for _, label in self.samples)
        inv_map = {v: k for k, v in self.class_map.items()}
        return {inv_map[k]: v for k, v in sorted(label_counts.items())}


# ──────────────────────────────────────────────────────────────────────
# Class weights
# ──────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset: StegoDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
    Returns a tensor of shape (num_classes,).
    """
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    n_total = len(labels)
    n_classes = len(config.CLASS_NAMES)

    weights = torch.zeros(n_classes)
    for cls_idx in range(n_classes):
        count = counts.get(cls_idx, 1)
        weights[cls_idx] = n_total / (n_classes * count)

    return weights


def get_weighted_sampler(dataset: StegoDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that oversamples minority classes.
    Helps with class imbalance at the batch level.
    """
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    n_classes = len(config.CLASS_NAMES)

    class_weight = {
        cls_idx: 1.0 / counts.get(cls_idx, 1)
        for cls_idx in range(n_classes)
    }
    sample_weights = [class_weight[label] for label in labels]
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )


# ──────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:           Path = None,
    batch_size:         int  = 32,
    num_workers:        int  = 4,
    use_noise_residual: bool = True,
    use_weighted_sampler: bool = True,   # NEW — oversamples minority classes
    input_size:         int  = 224,
):
    """
    Build train / val / test DataLoaders.

    Args:
        data_dir:             Override combined data directory
        batch_size:           Batch size
        num_workers:          DataLoader worker processes
        use_noise_residual:   Append Laplacian channel
        use_weighted_sampler: Use WeightedRandomSampler for training
        input_size:           Image size (default 224)

    Returns:
        train_loader, val_loader, test_loader, auto_class_weights
    """
    if data_dir is None:
        data_dir = config.COMBINED_DIR

    print("\nBuilding DataLoaders...")
    print(f"  Data dir:          {data_dir}")
    print(f"  Noise residual:    {use_noise_residual}")
    print(f"  Weighted sampler:  {use_weighted_sampler}")

    train_tf = _build_transforms("train", input_size)
    eval_tf  = _build_transforms("eval",  input_size)

    print("\nTrain dataset:")
    train_ds = StegoDataset(
        root_dir           = data_dir / "train",
        transform          = train_tf,
        use_noise_residual = use_noise_residual,
    )
    print("\nVal dataset:")
    val_ds   = StegoDataset(
        root_dir           = data_dir / "val",
        transform          = eval_tf,
        use_noise_residual = use_noise_residual,
    )
    print("\nTest dataset:")
    test_ds  = StegoDataset(
        root_dir           = data_dir / "test",
        transform          = eval_tf,
        use_noise_residual = use_noise_residual,
    )

    auto_class_weights = compute_class_weights(train_ds)
    print(f"\n  Auto class weights: {auto_class_weights.tolist()}")
    print(f"  Class counts: {train_ds.get_class_counts()}")

    # Sampler
    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_ds)
        shuffle = False   # sampler handles randomisation
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,   # avoid single-sample batches (BatchNorm issue)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader, auto_class_weights