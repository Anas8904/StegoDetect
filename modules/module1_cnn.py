"""
StegoDetect - Module 1: CNN Steganalysis Classifier

EfficientNet-B0 based classifier for 3-class steganography detection:
    Class 0: clean  (no hidden data)
    Class 1: lsb    (LSB steganography detected)
    Class 2: pvd    (PVD steganography detected)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class StegoClassifier(nn.Module):
    """
    EfficientNet-B0 based steganalysis classifier.

    Architecture:
        - EfficientNet-B0 backbone (pretrained on ImageNet)
        - Custom classification head with dropout
        - Optional 4-channel input for noise-residual features
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        use_noise_residual: bool = False,
        dropout: float = config.CNN_CONFIG["dropout"],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_noise_residual = use_noise_residual

        # Load EfficientNet-B0 backbone without classification head
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,        # Remove original head
            global_pool="avg",
        )

        # Feature dimension from backbone (1280 for EfficientNet-B0)
        feature_dim = self.backbone.num_features

        # Modify first conv if using noise-residual (4 channels)
        if use_noise_residual:
            original_conv = self.backbone.conv_stem
            new_conv = nn.Conv2d(
                4,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            # Copy pretrained weights for first 3 channels
            new_conv.weight.data[:, :3, :, :] = original_conv.weight.data
            # Initialize 4th channel weights to near-zero
            new_conv.weight.data[:, 3:, :, :] = 0.01 * torch.randn_like(
                new_conv.weight.data[:, 3:, :, :]
            )
            self.backbone.conv_stem = new_conv

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image tensor → class logits."""
        features = self.backbone(x)
        return self.classifier(features)

    def predict_with_confidence(self, x: torch.Tensor):
        """
        Run prediction and return class, confidence, and full probabilities.

        Returns:
            predicted_class: tensor of predicted class indices
            confidence: tensor of prediction confidences
            probs: tensor of shape (batch, num_classes) with all probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
        return predicted_class, confidence, probs


# ──────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────

def freeze_backbone_layers(model: StegoClassifier, num_layers_to_freeze: int = -1):
    """
    Freeze backbone layers for transfer learning.

    Args:
        model: StegoClassifier instance
        num_layers_to_freeze: Number of blocks to freeze.
                              -1 means freeze ALL backbone layers.
    """
    if num_layers_to_freeze == -1:
        # Freeze entire backbone
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


def get_model_summary(model: StegoClassifier):
    """Print model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"Model Summary:")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,}")
    print(f"  Frozen parameters:    {frozen_params:>12,}")
    print(f"  Model size:           {model_size_mb:>10.2f} MB")
    return total_params, trainable_params


def load_pretrained_stego(checkpoint_path: str, device=None):
    """
    Load a saved StegoClassifier checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Target device (default: config.DEVICE)

    Returns:
        Model in eval mode
    """
    if device is None:
        device = config.DEVICE

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine model config from checkpoint
    ckpt_config = checkpoint.get("config", {})
    num_classes = ckpt_config.get("num_classes", config.NUM_CLASSES)
    use_noise = ckpt_config.get("use_noise_residual", False)
    dropout = ckpt_config.get("dropout", config.CNN_CONFIG["dropout"])

    model = StegoClassifier(
        num_classes=num_classes,
        use_noise_residual=use_noise,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Val F1: {checkpoint.get('val_f1', '?')}")

    return model


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("StegoDetect - Module 1: CNN Steganalysis Classifier")
    print("=" * 50)

    device = config.DEVICE
    print(f"Device: {device}")

    # Create model
    model = StegoClassifier(
        num_classes=config.NUM_CLASSES,
        use_noise_residual=False,
        dropout=0.3,
    )
    model.to(device)

    # Print summary
    get_model_summary(model)

    # Test forward pass with random batch
    print("\nTest forward pass:")
    batch = torch.randn(4, 3, 224, 224).to(device)
    logits = model(batch)
    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {logits.shape}")

    pred_class, confidence, probs = model.predict_with_confidence(batch)
    print(f"  Predicted classes: {pred_class.tolist()}")
    print(f"  Confidences:       {[f'{c:.3f}' for c in confidence.tolist()]}")
    print(f"  Probabilities:\n{probs}")

    print("\n[OK] Module 1 CNN working correctly!")
