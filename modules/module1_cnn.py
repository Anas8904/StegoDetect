"""
StegoDetect - Module 1: CNN Steganalysis Classifier  (v2)

EfficientNet-B0 based classifier for 3-class steganography detection:
    Class 0: clean  (no hidden data)
    Class 1: lsb    (LSB steganography detected)
    Class 2: pvd    (PVD steganography detected)

CHANGES FROM v1:
  - SRMPreprocessor: fixed Spatial Rich Model high-pass kernels prepended
    before the backbone.  These 3 SRM filters are designed specifically to
    amplify the subtle noise patterns introduced by LSB steganography.
  - Improved classifier head: Linear(1280, 512) -> BN -> ReLU -> Dropout ->
    Linear(512, num_classes).  Gives the model more capacity to separate
    the three classes in feature space.
  - 4-channel backbone path unchanged (noise-residual), but now DEFAULT True.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────────────
# Spatial Rich Model (SRM) preprocessing layer
# ──────────────────────────────────────────────────────────────────────

class SRMPreprocessor(nn.Module):
    """
    Fixed high-pass filter layer inspired by the Spatial Rich Model (SRM)
    used in steganalysis.

    WHY THIS HELPS:
      LSB steganography modifies the least-significant bits of pixel values.
      After ImageNet normalisation these changes are at the ~0.001 scale and
      nearly invisible to a generic CNN feature extractor.  High-pass filters
      suppress the low-frequency image content (scene/texture) and amplify
      the high-frequency residual where the stego signal lives.

    Implementation:
      Three fixed (non-trainable) 3x3 kernels applied channel-wise.
      The filtered output is concatenated with the original RGB input,
      giving the backbone 6 channels when use_srm=True.
      (Or 7 channels when noise_residual 4th channel is ALSO enabled.)

    Note: We do NOT make these trainable — letting the model learn to
    undo the filters would defeat the purpose.
    """

    # Three canonical SRM kernels (values from Fridrich & Kodovsky 2012)
    KERNELS = [
        # Kernel 1 — simple Laplacian (detects pixel discontinuities)
        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]],

        # Kernel 2 — diagonal Laplacian
        [[-1,  0, -1],
         [ 0,  4,  0],
         [-1,  0, -1]],

        # Kernel 3 — 2nd-order horizontal gradient
        [[-1,  2, -1],
         [ 0,  0,  0],
         [ 1, -2,  1]],
    ]

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        num_kernels = len(self.KERNELS)

        # One output channel per SRM kernel, applied to EACH input channel
        # → out_channels = num_kernels * in_channels
        out_channels = num_kernels * in_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1,
            groups=in_channels,   # depthwise: each input ch processed separately
            bias=False,
        )

        # Fill with SRM weights — no gradient needed
        with torch.no_grad():
            for ch in range(in_channels):
                for k_idx, kernel in enumerate(self.KERNELS):
                    filter_idx = ch * num_kernels + k_idx
                    self.conv.weight[filter_idx, 0] = torch.tensor(
                        kernel, dtype=torch.float32
                    ) / 4.0   # normalise

        for param in self.conv.parameters():
            param.requires_grad = False   # FIXED — not learnable

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  — normalised input image
        Returns:
            residual: (B, num_kernels*C, H, W)  — SRM noise maps
        """
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────
# Main Classifier
# ──────────────────────────────────────────────────────────────────────

class StegoClassifier(nn.Module):
    """
    EfficientNet-B0 based steganalysis classifier (v2).

    Architecture:
        [optional] SRMPreprocessor  →  concatenate with original RGB
        EfficientNet-B0 backbone    (pretrained, modified conv_stem)
        Improved head: Linear(1280,512) → BN → ReLU → Dropout → Linear(512,C)
    """

    def __init__(
        self,
        num_classes: int           = config.NUM_CLASSES,
        use_noise_residual: bool   = config.CNN_CONFIG["use_noise_residual"],
        use_srm: bool              = True,    # NEW in v2
        dropout: float             = config.CNN_CONFIG["dropout"],
    ):
        super().__init__()
        self.num_classes       = num_classes
        self.use_noise_residual = use_noise_residual
        self.use_srm           = use_srm

        # ------------------------------------------------------------------
        # Compute number of input channels for the backbone conv_stem
        # ------------------------------------------------------------------
        # base RGB = 3 channels
        # + noise residual (Laplacian of RGB → 1 ch) if enabled
        # + SRM residual   (3 kernels × 3 RGB channels → 9 ch) if enabled
        srm_rgb_channels = 3   # SRM always operates on the original 3 RGB channels
        srm_out_channels = len(SRMPreprocessor.KERNELS) * srm_rgb_channels  # 9

        backbone_in_channels = 3  # always start with RGB
        if use_noise_residual:
            backbone_in_channels += 1   # Laplacian channel from dataset.py
        if use_srm:
            backbone_in_channels += srm_out_channels   # 9 SRM channels

        # ------------------------------------------------------------------
        # SRM module (fixed)
        # ------------------------------------------------------------------
        if use_srm:
            self.srm = SRMPreprocessor(in_channels=3)
        else:
            self.srm = None

        # ------------------------------------------------------------------
        # EfficientNet-B0 backbone
        # ------------------------------------------------------------------
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,       # remove head
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features   # 1280

        # Modify conv_stem if we're feeding more than 3 channels
        if backbone_in_channels != 3:
            orig_conv = self.backbone.conv_stem
            new_conv  = nn.Conv2d(
                backbone_in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            # Copy pretrained RGB weights for first 3 channels
            new_conv.weight.data[:, :3, :, :] = orig_conv.weight.data
            # Xavier-uniform init for extra channels (better than random)
            if backbone_in_channels > 3:
                nn.init.xavier_uniform_(new_conv.weight.data[:, 3:, :, :])
                # Scale down slightly so extra channels start small
                new_conv.weight.data[:, 3:, :, :] *= 0.05
            self.backbone.conv_stem = new_conv

        # ------------------------------------------------------------------
        # Improved classification head  (v2)
        # v1: Dropout → Linear(1280, C)
        # v2: Linear(1280,512) → BN → ReLU → Dropout → Linear(512, C)
        # The intermediate layer gives the model richer non-linear capacity
        # to separate the three classes in the learned feature space.
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Initialise the new linear layers properly
        nn.init.kaiming_normal_(self.classifier[0].weight, mode="fan_out",
                                nonlinearity="relu")
        nn.init.zeros_(self.classifier[0].bias)
        nn.init.normal_(self.classifier[4].weight, std=0.01)
        nn.init.zeros_(self.classifier[4].bias)

    # ──────────────────────────────────────────────────────────────────
    def _build_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Concatenate SRM channels and/or noise-residual channel with x.

        x may already contain the noise-residual 4th channel (added by
        dataset.py).  We extract only the first 3 RGB channels for SRM.
        """
        rgb = x[:, :3, :, :]   # always the first 3 channels

        parts = [x]             # start with full input (RGB + optional noise ch)
        if self.use_srm and self.srm is not None:
            parts.append(self.srm(rgb))

        if len(parts) == 1:
            return x
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image tensor → class logits."""
        x_in   = self._build_input(x)
        feats  = self.backbone(x_in)
        return self.classifier(feats)

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        thresholds: dict = None,
    ):
        """
        Run prediction and return class, confidence, and full probabilities.

        Args:
            x: input image tensor
            thresholds: optional per-class decision thresholds dict
                        {class_idx: float}.  If provided, a class is
                        predicted only when its probability exceeds its
                        threshold; otherwise falls back to argmax.
        Returns:
            predicted_class: tensor of predicted class indices
            confidence:      tensor of prediction confidences
            probs:           tensor (batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)

            if thresholds is not None:
                # Threshold-based decision (post-calibration)
                predicted_class = torch.argmax(probs, dim=1).clone()
                for cls_idx, thresh in thresholds.items():
                    # For each sample, if the argmax class probability is
                    # below threshold, fall back to the highest-prob class
                    mask = (probs[:, cls_idx] < thresh) & (predicted_class == cls_idx)
                    if mask.any():
                        # Exclude the low-confidence class and pick next best
                        alt_probs = probs.clone()
                        alt_probs[:, cls_idx] = -1.0
                        predicted_class[mask] = torch.argmax(alt_probs, dim=1)[mask]
                confidence = probs.gather(1, predicted_class.unsqueeze(1)).squeeze(1)
            else:
                confidence, predicted_class = torch.max(probs, dim=1)

        return predicted_class, confidence, probs


# ──────────────────────────────────────────────────────────────────────
# Utility functions  (unchanged API, compatible with v1 checkpoints)
# ──────────────────────────────────────────────────────────────────────

def freeze_backbone_layers(model: StegoClassifier, num_layers_to_freeze: int = -1):
    """Freeze backbone layers for transfer learning."""
    if num_layers_to_freeze == -1:
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        blocks = list(model.backbone.blocks)
        for i, block in enumerate(blocks):
            if i < num_layers_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False


def unfreeze_all(model: StegoClassifier):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    # Keep SRM fixed regardless
    if model.srm is not None:
        for param in model.srm.parameters():
            param.requires_grad = False


def get_model_summary(model: StegoClassifier):
    """Print model parameter summary."""
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    model_size_mb    = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    print("Model Summary:")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,}")
    print(f"  Frozen parameters:    {frozen_params:>12,}")
    print(f"  Model size:           {model_size_mb:>10.2f} MB")
    print(f"  Noise residual:       {model.use_noise_residual}")
    print(f"  SRM preprocessing:    {model.use_srm}")
    return total_params, trainable_params


def load_pretrained_stego(checkpoint_path: str, device=None):
    """Load a saved StegoClassifier checkpoint."""
    if device is None:
        device = config.DEVICE

    checkpoint  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", {})

    model = StegoClassifier(
        num_classes        = ckpt_config.get("num_classes",        config.NUM_CLASSES),
        use_noise_residual = ckpt_config.get("use_noise_residual", True),
        use_srm            = ckpt_config.get("use_srm",            True),
        dropout            = ckpt_config.get("dropout",            config.CNN_CONFIG["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch:  {checkpoint.get('epoch', '?')}")
    print(f"  Val F1: {checkpoint.get('val_f1', '?')}")
    return model


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("StegoDetect - Module 1: CNN Steganalysis Classifier (v2)")
    print("=" * 60)

    device = config.DEVICE
    print(f"Device: {device}")

    model = StegoClassifier(
        num_classes        = config.NUM_CLASSES,
        use_noise_residual = True,
        use_srm            = True,
        dropout            = 0.4,
    )
    model.to(device)
    get_model_summary(model)

    # Test with 4-channel input (RGB + noise residual)
    print("\nTest forward pass (4-ch input + SRM):")
    model.eval()
    batch  = torch.randn(4, 4, 224, 224).to(device)   # 4-ch: RGB + Laplacian
    logits = model(batch)
    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {logits.shape}")

    pred_class, confidence, probs = model.predict_with_confidence(batch)
    print(f"  Predicted classes: {pred_class.tolist()}")
    print(f"  Confidences:       {[f'{c:.3f}' for c in confidence.tolist()]}")
    print("[OK] Module 1 v2 CNN working correctly!")