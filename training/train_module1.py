"""
StegoDetect - Training Script for Module 1 (EfficientNet CNN)  v3

CHANGES FROM v2:
  - Linear LR warmup for Phase 2 (warmup_epochs=3).
    Fixes the val_loss=0.939 spike seen at epoch 40 in v2 logs.
  - dataset.py now uses SAFE augmentation (no color jitter).
    This is the most impactful fix — color jitter was erasing the
    LSB signal before the model ever saw it.
  - WeightedRandomSampler support in get_dataloaders().
  - Updated to use config.CNN_CONFIG["use_srm"] (default False in v3).
  - Gradient accumulation support (accumulate_steps arg) for
    effective large-batch training on low-VRAM GPUs.
  - Per-class accuracy printed each epoch so you can track LSB recall.

Usage:
    # Standard run (recommended first try):
    python training/train_module1.py

    # With gradient accumulation (effective batch = 32×4 = 128):
    python training/train_module1.py --accumulate_steps 4

    # Resume from last checkpoint:
    python training/train_module1.py --resume

    # Low-VRAM mode (use B0 in config.py, smaller batch):
    python training/train_module1.py --batch_size 16 --accumulate_steps 2
"""

import os
import sys
import csv
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from modules.module1_cnn import (
    StegoClassifier,
    freeze_backbone_layers,
    unfreeze_all,
    get_model_summary,
)
from modules.dataset import get_dataloaders
from data_prep.decode_images import decode_test_images


# ──────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with per-class weighting and label smoothing.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2.0 down-weights easy correct predictions (clean, pvd) and
    concentrates gradients on the hard LSB examples.
    """

    def __init__(
        self,
        weight: torch.Tensor = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            inputs, targets,
            weight          = self.weight,
            reduction       = "none",
            label_smoothing = self.label_smoothing,
        )
        with torch.no_grad():
            probs = F.softmax(inputs, dim=1)
            p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────────────────────────────
# Warmup LR scheduler
# ──────────────────────────────────────────────────────────────────────

class LinearWarmupCosineAnnealingLR:
    """
    Linear warmup for `warmup_epochs`, then cosine annealing.

    Wraps two PyTorch schedulers.  Call `.step()` once per epoch.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-7):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.eta_min        = eta_min
        self.current_epoch  = 0
        self.base_lrs       = [pg["lr"] for pg in optimizer.param_groups]

        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max   = max(1, total_epochs - warmup_epochs),
            eta_min = eta_min,
        )

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup: scale from 0 → base_lr over warmup_epochs
            scale = self.current_epoch / max(1, self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * scale
        else:
            self.cosine_scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ──────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    max_grad_norm=1.0, accumulate_steps=1,
):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    use_amp = (device.type == "cuda")
    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader), desc="  Train", leave=False)
    for step, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
            loss   = loss / accumulate_steps   # scale for accumulation

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulate_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()

        loss_item = loss.item() * accumulate_steps
        running_loss += loss_item * images.size(0)
        _, predicted  = torch.max(logits, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(
            loss=f"{loss_item:.4f}",
            acc=f"{100*correct/total:.1f}%",
        )

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model. Returns loss, accuracy, preds, labels, probs."""
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []
    all_probs    = []

    use_amp = (device.type == "cuda")

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs         = torch.softmax(logits, dim=1)
        _, predicted  = torch.max(probs, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        running_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def print_per_class_stats(labels, preds, class_names):
    """Print per-class accuracy to track LSB recall specifically."""
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    for i, name in enumerate(class_names):
        print(f"           {name:>6}: P={precision[i]:.3f}  "
              f"R={recall[i]:.3f}  F1={f1_per_class[i]:.3f}  "
              f"n={support[i]}")


# ──────────────────────────────────────────────────────────────────────
# Threshold calibration
# ──────────────────────────────────────────────────────────────────────

def calibrate_thresholds(val_labels, val_probs, target_fpr: float = 0.02):
    """
    Find the optimal clean-class threshold on the validation set.
    Returns {0: threshold} or None if no improvement.
    """
    best_f1     = -1.0
    best_thresh = None

    val_labels = np.array(val_labels)
    val_probs  = np.array(val_probs)

    for t in np.arange(0.30, 0.91, 0.01):
        preds = np.argmax(val_probs, axis=1).copy()
        clean_pred_mask = preds == 0
        low_conf_mask   = val_probs[:, 0] < t
        switch_mask     = clean_pred_mask & low_conf_mask

        if switch_mask.sum() > 0:
            alt_probs       = val_probs.copy()
            alt_probs[:, 0] = -1.0
            alt_preds       = np.argmax(alt_probs, axis=1)
            preds[switch_mask] = alt_preds[switch_mask]

        macro_f1  = f1_score(val_labels, preds, average="macro", zero_division=0)
        clean_mask = val_labels == 0
        fpr_clean  = ((preds[clean_mask] != 0).sum() / clean_mask.sum()
                      if clean_mask.sum() > 0 else 0.0)

        if fpr_clean <= target_fpr and macro_f1 > best_f1:
            best_f1     = macro_f1
            best_thresh = t

    if best_thresh is not None:
        print(f"  Calibration: clean threshold = {best_thresh:.2f}  "
              f"(val macro-F1={best_f1:.4f})")
        return {0: float(best_thresh)}

    print("  Calibration: no improvement found — using argmax (threshold=0.5)")
    return None


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, save_path, epoch, class_names):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_roc_curves(y_true, y_probs, save_path, class_names):
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5))
    if len(class_names) == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, class_names)):
        y_bin    = (np.array(y_true) == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        auc_val  = roc_auc_score(y_bin, y_probs[:, i])
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(f"ROC - {name}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_training_curves(log_path, save_path):
    import csv as _csv
    epochs, train_loss, val_loss, val_f1 = [], [], [], []
    try:
        with open(log_path) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                val_f1.append(float(row["val_f1_macro"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, train_loss, label="Train Loss")
        ax1.plot(epochs, val_loss,   label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()

        ax2.plot(epochs, val_f1, label="Val Macro F1", color="green")
        ax2.axhline(0.80, color="red", linestyle="--", alpha=0.6, label="80% target")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro F1")
        ax2.set_title("Validation F1")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to {save_path}")
    except Exception as e:
        print(f"  [WARN] Could not save training curves: {e}")


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if args.device != "auto" else str(config.DEVICE))
    use_srm = config.CNN_CONFIG.get("use_srm", False)

    print(f"\n{'='*60}")
    print(f"StegoDetect - Module 1 Training  (v3)")
    print(f"{'='*60}")
    print(f"Device:          {device}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size} (effective: {args.batch_size * args.accumulate_steps})")
    print(f"Accumulate:      {args.accumulate_steps} steps")
    print(f"Noise residual:  {args.use_noise_residual}")
    print(f"SRM layer:       {use_srm}")
    print(f"Focal loss:      {config.CNN_CONFIG['use_focal_loss']}")
    print(f"Warmup epochs:   {config.CNN_CONFIG['warmup_epochs']}")
    print(f"Resume:          {args.resume}")
    print(f"{'='*60}\n")

    # ── Decrypt test images ───────────────────────────────────────────
    print("Decrypting test images if necessary...")
    decode_test_images(str(config.COMBINED_DIR / "test" / "lsb"))

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_loader, val_loader, test_loader, auto_class_weights = get_dataloaders(
        batch_size            = args.batch_size,
        num_workers           = args.num_workers,
        use_noise_residual    = args.use_noise_residual,
        use_weighted_sampler  = config.CNN_CONFIG.get("use_weighted_sampler", True),
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # ── Class weights ─────────────────────────────────────────────────
    manual_w = config.CNN_CONFIG["class_weights_manual"]
    class_weights = torch.tensor(manual_w, dtype=torch.float32)
    print(f"  Manual class weights: {class_weights.tolist()}")

    # ── Model ─────────────────────────────────────────────────────────
    model = StegoClassifier(
        num_classes        = config.NUM_CLASSES,
        use_noise_residual = args.use_noise_residual,
        use_srm            = use_srm,
        dropout            = config.CNN_CONFIG["dropout"],
    ).to(device)
    get_model_summary(model)

    # ── Loss ──────────────────────────────────────────────────────────
    if config.CNN_CONFIG.get("use_focal_loss", True):
        criterion = FocalLoss(
            weight          = class_weights.to(device),
            gamma           = config.CNN_CONFIG["focal_gamma"],
            label_smoothing = config.CNN_CONFIG["label_smoothing"],
        )
        print(f"Loss: FocalLoss (gamma={config.CNN_CONFIG['focal_gamma']}, "
              f"ls={config.CNN_CONFIG['label_smoothing']})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight          = class_weights.to(device),
            label_smoothing = config.CNN_CONFIG.get("label_smoothing", 0.0),
        )
        print("Loss: CrossEntropyLoss with label smoothing")

    # ── Mixed-precision scaler ────────────────────────────────────────
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # ── Checkpoint dirs ───────────────────────────────────────────────
    ckpt_dir = config.MODULE1_CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV log ───────────────────────────────────────────────────────
    log_path   = config.CHECKPOINTS_DIR / "training_log.csv"
    log_fields = [
        "epoch", "phase", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_f1_macro", "val_f1_weighted",
        "lr", "time_sec",
    ]

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch      = 0
    best_val_f1      = 0.0
    patience_counter = 0

    if args.resume:
        last_path = ckpt_dir / "last_model.pth"
        if last_path.exists():
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch      = ckpt["epoch"]
            best_val_f1      = ckpt.get("best_val_f1", 0.0)
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"Resumed from epoch {start_epoch} (best F1: {best_val_f1:.4f})")
        else:
            print(f"[WARN] No checkpoint at {last_path}, starting fresh.")

    # ── Open log file ─────────────────────────────────────────────────
    log_mode   = "a" if (args.resume and log_path.exists()) else "w"
    log_file   = open(log_path, log_mode, newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if log_mode == "w":
        log_writer.writeheader()

    # ──────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ──────────────────────────────────────────────────────────────────
    phase1_epochs  = config.CNN_CONFIG["phase1_epochs"]
    warmup_epochs  = config.CNN_CONFIG.get("warmup_epochs", 3)
    total_epochs   = args.epochs
    scheduler      = None

    for epoch in range(start_epoch, total_epochs):
        epoch_start   = time.time()
        current_epoch = epoch + 1

        # ── Phase ─────────────────────────────────────────────────────
        if current_epoch <= phase1_epochs:
            phase     = 1
            phase_str = "Phase 1 (frozen backbone)"
        else:
            phase     = 2
            phase_str = "Phase 2 (full fine-tuning)"

        # ── Optimizer (rebuild at phase transitions) ───────────────────
        if current_epoch == 1 or (current_epoch == phase1_epochs + 1):
            if phase == 1:
                freeze_backbone_layers(model, num_layers_to_freeze=-1)
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr           = config.CNN_CONFIG["phase1_lr"],
                    weight_decay = config.CNN_CONFIG["weight_decay"],
                )
                remaining = phase1_epochs - epoch
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max   = max(1, remaining),
                    eta_min = config.CNN_CONFIG["scheduler_eta_min"],
                )
            else:
                # Phase 2: unfreeze, differential LR, warmup
                unfreeze_all(model)
                optimizer = AdamW([
                    {
                        "params": model.backbone.parameters(),
                        "lr":     config.CNN_CONFIG["phase2_lr_backbone"],
                    },
                    {
                        "params": model.classifier.parameters(),
                        "lr":     config.CNN_CONFIG["phase2_lr_classifier"],
                    },
                ], weight_decay=config.CNN_CONFIG["weight_decay"])

                remaining = total_epochs - epoch
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs = warmup_epochs,
                    total_epochs  = remaining,
                    eta_min       = config.CNN_CONFIG["scheduler_eta_min"],
                )
                print(f"\n  [Phase 2] LR warmup for {warmup_epochs} epochs then cosine")

        # ── Train ─────────────────────────────────────────────────────
        print(f"\nEpoch {current_epoch}/{total_epochs} — {phase_str}")
        train_loss, train_acc, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            max_grad_norm    = config.CNN_CONFIG["gradient_clip_max_norm"],
            accumulate_steps = args.accumulate_steps,
        )

        # ── Validate ──────────────────────────────────────────────────
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device,
        )

        # ── Metrics ───────────────────────────────────────────────────
        val_f1_macro    = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_f1_weighted = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        current_lr  = optimizer.param_groups[0]["lr"]
        epoch_time  = time.time() - epoch_start

        print(f"  Train  — Loss: {train_loss:.4f}  Acc: {100*train_acc:.2f}%")
        print(f"  Val    — Loss: {val_loss:.4f}  Acc: {100*val_acc:.2f}%  "
              f"F1(macro): {val_f1_macro:.4f}  F1(weighted): {val_f1_weighted:.4f}")
        print_per_class_stats(val_labels, val_preds, config.CLASS_NAMES)
        print(f"  LR: {current_lr:.2e}  Time: {epoch_time:.1f}s")

        # ── Log ───────────────────────────────────────────────────────
        log_writer.writerow({
            "epoch":          current_epoch,
            "phase":          phase,
            "train_loss":     f"{train_loss:.6f}",
            "train_acc":      f"{train_acc:.6f}",
            "val_loss":       f"{val_loss:.6f}",
            "val_acc":        f"{val_acc:.6f}",
            "val_f1_macro":   f"{val_f1_macro:.6f}",
            "val_f1_weighted":f"{val_f1_weighted:.6f}",
            "lr":             f"{current_lr:.2e}",
            "time_sec":       f"{epoch_time:.1f}",
        })
        log_file.flush()

        # ── Confusion matrix every 5 epochs ───────────────────────────
        if current_epoch % 5 == 0 or current_epoch == total_epochs:
            save_confusion_matrix(
                val_labels, val_preds,
                ckpt_dir / f"confusion_matrix_epoch_{current_epoch}.png",
                current_epoch, config.CLASS_NAMES,
            )

        # ── Checkpoint ────────────────────────────────────────────────
        ckpt_data = {
            "epoch":               current_epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "val_f1":              val_f1_macro,
            "val_acc":             val_acc,
            "val_loss":            val_loss,
            "best_val_f1":         best_val_f1,
            "patience_counter":    patience_counter,
            "config": {
                "num_classes":        config.NUM_CLASSES,
                "use_noise_residual": args.use_noise_residual,
                "use_srm":            use_srm,
                "dropout":            config.CNN_CONFIG["dropout"],
                "batch_size":         args.batch_size,
                "epochs":             args.epochs,
            },
        }
        torch.save(ckpt_data, ckpt_dir / "last_model.pth")

        if current_epoch % 10 == 0:
            torch.save(ckpt_data, ckpt_dir / f"checkpoint_epoch_{current_epoch}.pth")

        if val_f1_macro > best_val_f1:
            best_val_f1      = val_f1_macro
            patience_counter = 0
            torch.save(ckpt_data, ckpt_dir / "best_model.pth")
            print(f"  ★ New best model saved! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement "
                  f"({patience_counter}/{config.CNN_CONFIG['early_stopping_patience']})")

        if scheduler is not None:
            scheduler.step()

        if patience_counter >= config.CNN_CONFIG["early_stopping_patience"]:
            print(f"\n[STOP] Early stopping at epoch {current_epoch}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_file.close()

    # ── Training curves ───────────────────────────────────────────────
    save_training_curves(
        log_path,
        ckpt_dir / "training_curves.png",
    )

    # ──────────────────────────────────────────────────────────────────
    # THRESHOLD CALIBRATION  (on val set)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("THRESHOLD CALIBRATION")
    print(f"{'='*60}")

    best_path = ckpt_dir / "best_model.pth"
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    print(f"Calibrating on val set using best model (epoch {best_ckpt['epoch']})...")

    _, _, cal_preds, cal_labels, cal_probs = validate(model, val_loader, criterion, device)
    optimal_thresholds = calibrate_thresholds(cal_labels, cal_probs, target_fpr=0.02)

    thresh_path = ckpt_dir / "decision_thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(
            {str(k): v for k, v in (optimal_thresholds or {}).items()},
            f, indent=2,
        )
    print(f"Thresholds saved to {thresh_path}")

    # ──────────────────────────────────────────────────────────────────
    # FINAL EVALUATION ON TEST SET
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    print(f"Best model from epoch {best_ckpt['epoch']} "
          f"(Val F1: {best_ckpt['val_f1']:.4f})")

    test_loss, test_acc, test_preds_raw, test_labels, test_probs = validate(
        model, test_loader, criterion, device,
    )

    test_preds = test_preds_raw.copy()
    if optimal_thresholds:
        for cls_idx, thresh in optimal_thresholds.items():
            clean_pred_mask = test_preds == cls_idx
            low_conf_mask   = test_probs[:, cls_idx] < thresh
            switch_mask     = clean_pred_mask & low_conf_mask
            if switch_mask.sum() > 0:
                alt_probs           = test_probs.copy()
                alt_probs[:, cls_idx] = -1.0
                alt_preds           = np.argmax(alt_probs, axis=1)
                test_preds[switch_mask] = alt_preds[switch_mask]

    report = classification_report(
        test_labels, test_preds,
        target_names=config.CLASS_NAMES,
        digits=4,
    )
    print(f"\nTest Accuracy: {100*test_acc:.2f}%")
    print(f"\nClassification Report:\n{report}")

    save_confusion_matrix(
        test_labels, test_preds,
        ckpt_dir / "final_confusion_matrix.png",
        "Final", config.CLASS_NAMES,
    )

    try:
        save_roc_curves(
            test_labels, test_probs,
            ckpt_dir / "roc_curves.png",
            config.CLASS_NAMES,
        )
    except Exception as e:
        print(f"[WARN] ROC curves failed: {e}")

    clean_mask  = test_labels == 0
    clean_total = clean_mask.sum()
    fpr_clean   = 0.0
    if clean_total > 0:
        fp_clean  = ((test_preds != 0) & clean_mask).sum()
        fpr_clean = fp_clean / clean_total
        print(f"\nClean Image False Positive Rate: {100*fpr_clean:.2f}%")
        status = "[OK]" if fpr_clean <= 0.02 else "[WARN]"
        print(f"  {status} Target: ≤2%")

    test_metrics = {
        "test_accuracy":        float(test_acc),
        "test_loss":            float(test_loss),
        "best_epoch":           int(best_ckpt["epoch"]),
        "best_val_f1":          float(best_ckpt["val_f1"]),
        "classification_report":report,
        "clean_fpr":            float(fpr_clean),
        "optimal_thresholds":   optimal_thresholds,
        "timestamp":            datetime.now().isoformat(),
    }
    metrics_path = config.RESULTS_DIR / "module1_eval.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Test metrics saved to {metrics_path}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train StegoDetect Module 1 CNN  (v3)")
    parser.add_argument("--epochs",          type=int, default=config.CNN_CONFIG["epochs"])
    parser.add_argument("--batch_size",      type=int, default=config.CNN_CONFIG["batch_size"])
    parser.add_argument("--num_workers",     type=int, default=config.CNN_CONFIG["num_workers"])
    parser.add_argument("--device",          type=str, default="auto")
    parser.add_argument("--accumulate_steps",type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size × steps)")
    parser.add_argument(
        "--use_noise_residual", action="store_true",
        default=config.CNN_CONFIG["use_noise_residual"],
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)