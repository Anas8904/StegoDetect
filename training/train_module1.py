"""
StegoDetect - Training Script for Module 1 (EfficientNet-B4 CNN)

Two-phase training strategy with modern optimizations:
    Phase 1 (Epochs 1-5):   Frozen backbone, train classifier head (lr=5e-4)
    Phase 2 (Epochs 6-50):  Unfreeze all, differential lr (backbone=5e-5, head=5e-4)

Enhancements over B0 baseline:
    - Mixed precision training (AMP) for speed + memory savings
    - Linear LR warmup for training stability
    - Label smoothing (0.1) to prevent overconfident predictions
    - Higher dropout (0.4) for B4's larger capacity
    - CosineAnnealing with lower eta_min

Usage:
    python training/train_module1.py --epochs 50 --batch_size 16 --device cuda
    python training/train_module1.py --resume --device cuda
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from modules.module1_cnn import (
    StegoClassifier,
    freeze_backbone_layers,
    unfreeze_all,
    get_model_summary,
)
from modules.dataset import get_dataloaders


# ──────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp,
                    max_grad_norm=1.0):
    """Train for one epoch with optional mixed precision. Returns avg loss, accuracy, preds, labels."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # Scaled backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    """Validate model. Returns loss, accuracy, preds, labels, probs."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits.float(), dim=1)
        _, predicted = torch.max(probs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def save_confusion_matrix(y_true, y_pred, save_path, epoch, class_names):
    """Save confusion matrix as a heatmap PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_roc_curves(y_true, y_probs, save_path, class_names):
    """Save ROC curves for each class (one-vs-rest)."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5))
    if len(class_names) == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, class_names)):
        y_bin = (np.array(y_true) == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        auc_val = roc_auc_score(y_bin, y_probs[:, i])
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
    """Plot training curves from CSV log."""
    try:
        import pandas as pd
        df = pd.read_csv(log_path)
        if len(df) < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='.')
        axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='.')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[1].plot(df['epoch'], df['train_acc'].astype(float) * 100, label='Train Acc', marker='.')
        axes[1].plot(df['epoch'], df['val_acc'].astype(float) * 100, label='Val Acc', marker='.')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # F1 curves
        axes[2].plot(df['epoch'], df['val_f1_macro'], label='F1 Macro', marker='.')
        axes[2].plot(df['epoch'], df['val_f1_weighted'], label='F1 Weighted', marker='.')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('F1 Score Curves')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] Could not plot training curves: {e}")


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────

def train(args):
    """Main training function."""
    device = torch.device(args.device if args.device != "auto" else str(config.DEVICE))
    use_amp = config.CNN_CONFIG.get("use_amp", True) and device.type == "cuda"

    print(f"\n{'='*60}")
    print(f"StegoDetect -- Module 1 Training (EfficientNet-B4)")
    print(f"{'='*60}")
    print(f"Device:          {device}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Input size:      {config.CNN_CONFIG['input_size']}x{config.CNN_CONFIG['input_size']}")
    print(f"Mixed precision: {use_amp}")
    print(f"Label smoothing: {config.CNN_CONFIG.get('label_smoothing', 0.0)}")
    print(f"Noise residual:  {args.use_noise_residual}")
    print(f"Resume:          {args.resume}")
    print(f"{'='*60}\n")

    # ── Decode Test Images ────────────────────────────────────────────
    try:
        from data_prep.decode_images import decode_test_images
        print("Decoding test images (b64/zip) if necessary...")
        decode_test_images(str(config.COMBINED_DIR / "test" / "lsb"))
    except ImportError:
        print("[INFO] decode_images not available, skipping test image decoding")

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_noise_residual=args.use_noise_residual,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Class weights: {class_weights.tolist()}")

    # ── Model ─────────────────────────────────────────────────────────
    model = StegoClassifier(
        num_classes=config.NUM_CLASSES,
        use_noise_residual=args.use_noise_residual,
        dropout=config.CNN_CONFIG["dropout"],
    ).to(device)
    get_model_summary(model)

    # ── Loss with label smoothing ─────────────────────────────────────
    label_smoothing = config.CNN_CONFIG.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )

    # ── AMP Scaler ────────────────────────────────────────────────────
    scaler = GradScaler(enabled=use_amp)

    # ── Checkpoint dirs ───────────────────────────────────────────────
    ckpt_dir = config.MODULE1_CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV training log ──────────────────────────────────────────────
    log_path = config.CHECKPOINTS_DIR / "training_log.csv"
    log_fields = [
        "epoch", "phase", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_f1_macro", "val_f1_weighted",
        "lr", "time_sec",
    ]

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0

    if args.resume:
        last_path = ckpt_dir / "last_model.pth"
        if last_path.exists():
            checkpoint = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_f1 = checkpoint.get("best_val_f1", 0.0)
            patience_counter = checkpoint.get("patience_counter", 0)
            print(f"\n[OK] Resumed from epoch {start_epoch} (best F1: {best_val_f1:.4f})")
        else:
            print(f"\n[WARN] No checkpoint found at {last_path}, starting fresh.")

    # ── Training log CSV (append mode if resuming) ────────────────────
    log_mode = "a" if args.resume and log_path.exists() else "w"
    log_file = open(log_path, log_mode, newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if log_mode == "w":
        log_writer.writeheader()

    # ──────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ──────────────────────────────────────────────────────────────────
    phase1_epochs = config.CNN_CONFIG["phase1_epochs"]
    warmup_epochs = config.CNN_CONFIG.get("warmup_epochs", 3)
    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        current_epoch = epoch + 1  # 1-indexed for display

        # ── Determine training phase ──────────────────────────────────
        if current_epoch <= phase1_epochs:
            phase = 1
            phase_str = f"Phase 1 (frozen backbone)"
        else:
            phase = 2
            phase_str = f"Phase 2 (full fine-tuning)"

        # ── Set up optimizer for current phase ────────────────────────
        if current_epoch == 1 or (current_epoch == phase1_epochs + 1):
            if phase == 1:
                freeze_backbone_layers(model, num_layers_to_freeze=-1)
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config.CNN_CONFIG["phase1_lr"],
                    weight_decay=config.CNN_CONFIG["weight_decay"],
                )
            else:
                unfreeze_all(model)
                optimizer = AdamW([
                    {"params": model.backbone.parameters(), "lr": config.CNN_CONFIG["phase2_lr_backbone"]},
                    {"params": model.classifier.parameters(), "lr": config.CNN_CONFIG["phase2_lr_classifier"]},
                ], weight_decay=config.CNN_CONFIG["weight_decay"])

            remaining_epochs = total_epochs - epoch

            # Build scheduler: LinearLR warmup -> CosineAnnealing
            warmup_start_factor = config.CNN_CONFIG.get("warmup_start_factor", 0.01)
            actual_warmup = min(warmup_epochs, remaining_epochs)

            if actual_warmup > 0 and remaining_epochs > actual_warmup:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    total_iters=actual_warmup,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=remaining_epochs - actual_warmup,
                    eta_min=config.CNN_CONFIG["scheduler_eta_min"],
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[actual_warmup],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(remaining_epochs, 1),
                    eta_min=config.CNN_CONFIG["scheduler_eta_min"],
                )

        # ── Train ─────────────────────────────────────────────────────
        print(f"\nEpoch {current_epoch}/{total_epochs} -- {phase_str}")
        train_loss, train_acc, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp,
            max_grad_norm=config.CNN_CONFIG["gradient_clip_max_norm"],
        )

        # ── Validate on validation set ─────────────────────────────────
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device, use_amp=use_amp,
        )

        # ── Metrics ───────────────────────────────────────────────────
        val_f1_macro = f1_score(val_labels, val_preds, average="macro")
        val_f1_weighted = f1_score(val_labels, val_preds, average="weighted")

        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None, labels=[0, 1, 2],
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # ── Print summary ─────────────────────────────────────────────
        print(f"  Train -- Loss: {train_loss:.4f}  Acc: {100*train_acc:.2f}%")
        print(f"  Val   -- Loss: {val_loss:.4f}  Acc: {100*val_acc:.2f}%  "
              f"F1(macro): {val_f1_macro:.4f}  F1(weighted): {val_f1_weighted:.4f}")
        for i, name in enumerate(config.CLASS_NAMES):
            print(f"         {name:>6}: P={precision[i]:.3f}  R={recall[i]:.3f}  F1={f1_per_class[i]:.3f}")
        print(f"  LR: {current_lr:.2e}  Time: {epoch_time:.1f}s  AMP: {use_amp}")

        # ── Log to CSV ────────────────────────────────────────────────
        log_writer.writerow({
            "epoch": current_epoch,
            "phase": phase,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "val_f1_macro": f"{val_f1_macro:.6f}",
            "val_f1_weighted": f"{val_f1_weighted:.6f}",
            "lr": f"{current_lr:.2e}",
            "time_sec": f"{epoch_time:.1f}",
        })
        log_file.flush()

        # ── Save confusion matrix every 5 epochs ─────────────────────
        if current_epoch % 5 == 0 or current_epoch == total_epochs:
            cm_path = ckpt_dir / f"confusion_matrix_epoch_{current_epoch}.png"
            save_confusion_matrix(val_labels, val_preds, cm_path, current_epoch, config.CLASS_NAMES)

        # ── Checkpointing ─────────────────────────────────────────────
        checkpoint_data = {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1_macro,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "best_val_f1": best_val_f1,
            "patience_counter": patience_counter,
            "config": {
                "model_name": config.CNN_CONFIG["model_name"],
                "num_classes": config.NUM_CLASSES,
                "use_noise_residual": args.use_noise_residual,
                "dropout": config.CNN_CONFIG["dropout"],
                "input_size": config.CNN_CONFIG["input_size"],
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "label_smoothing": label_smoothing,
            },
        }

        # Save last (always)
        torch.save(checkpoint_data, ckpt_dir / "last_model.pth")

        # Save periodic checkpoints every 10 epochs
        if current_epoch % 10 == 0:
            torch.save(checkpoint_data, ckpt_dir / f"checkpoint_epoch_{current_epoch}.pth")

        # Save best model (based on val F1)
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            torch.save(checkpoint_data, ckpt_dir / "best_model.pth")
            print(f"  >> New best model saved! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.CNN_CONFIG['early_stopping_patience']})")

        # ── Step scheduler ────────────────────────────────────────────
        scheduler.step()

        # ── Early stopping ────────────────────────────────────────────
        if patience_counter >= config.CNN_CONFIG["early_stopping_patience"]:
            print(f"\n[WARN] Early stopping triggered at epoch {current_epoch}")
            break

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_file.close()

    # ──────────────────────────────────────────────────────────────────
    # TRAINING CURVES
    # ──────────────────────────────────────────────────────────────────
    curves_path = ckpt_dir / "training_curves.png"
    save_training_curves(log_path, curves_path)
    if curves_path.exists():
        print(f"Training curves saved to {curves_path}")

    # ──────────────────────────────────────────────────────────────────
    # FINAL EVALUATION ON TEST SET
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")

    # Load best model
    best_path = ckpt_dir / "best_model.pth"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()
        print(f"Loaded best model from epoch {best_ckpt['epoch']} (Val F1: {best_ckpt['val_f1']:.4f})")
    else:
        print("[WARN] No best model found, using last epoch weights")

    # Use criterion without label smoothing for clean evaluation
    eval_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, eval_criterion, device, use_amp=use_amp,
    )

    # Classification report
    report = classification_report(
        test_labels, test_preds,
        target_names=config.CLASS_NAMES,
        digits=4,
    )
    print(f"\nTest Accuracy: {100*test_acc:.2f}%")
    print(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm_path = ckpt_dir / "final_confusion_matrix.png"
    save_confusion_matrix(test_labels, test_preds, cm_path, "Final", config.CLASS_NAMES)
    print(f"Confusion matrix saved to {cm_path}")

    # ROC curves
    roc_path = ckpt_dir / "roc_curves.png"
    try:
        save_roc_curves(test_labels, test_probs, roc_path, config.CLASS_NAMES)
        print(f"ROC curves saved to {roc_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate ROC curves: {e}")

    # FALSE POSITIVE RATE for clean images
    clean_mask = test_labels == 0
    clean_total = clean_mask.sum()
    if clean_total > 0:
        false_positive_clean = ((test_preds != 0) & clean_mask).sum()
        fpr_clean = false_positive_clean / clean_total
        print(f"\nClean Image False Positive Rate: {100*fpr_clean:.2f}%")
        if fpr_clean > 0.02:
            print(f"  [WARN] FPR ({100*fpr_clean:.2f}%) exceeds 2% target!")
        else:
            print(f"  [OK] FPR is within the 2% target")

    # Save test metrics
    test_metrics = {
        "model": config.CNN_CONFIG["model_name"],
        "input_size": config.CNN_CONFIG["input_size"],
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "best_epoch": int(best_ckpt["epoch"]) if best_path.exists() else -1,
        "best_val_f1": float(best_ckpt.get("val_f1", 0)) if best_path.exists() else 0,
        "classification_report": report,
        "clean_fpr": float(fpr_clean) if clean_total > 0 else None,
        "label_smoothing": label_smoothing,
        "use_amp": use_amp,
        "timestamp": datetime.now().isoformat(),
    }
    metrics_path = config.RESULTS_DIR / "module1_eval.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"\nTest metrics saved to {metrics_path}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train StegoDetect Module 1 CNN (EfficientNet-B4)")
    parser.add_argument("--epochs", type=int, default=config.CNN_CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.CNN_CONFIG["batch_size"],
                        help="Batch size (default: 16 for B4)")
    parser.add_argument("--num_workers", type=int, default=config.CNN_CONFIG["num_workers"],
                        help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--use_noise_residual", action="store_true",
                        help="Use noise-residual 4th channel")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
