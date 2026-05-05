"""
StegoDetect - Project Configuration  (v3)

CHANGES FROM v2:
  KEY BUG FIX:
  - dataset.py now uses SAFE augmentation only (no color jitter).
    Color jitter was modifying pixel values and ERASING the LSB signal.
    This is likely the primary reason accuracy was stuck at 68%.

  HYPERPARAMETER CHANGES:
  - phase1_epochs: 10     (was 5)  — head needs more warm-up
  - phase2_lr_backbone: 2e-5 (was 5e-5) — fixes the epoch-40 val_loss spike
  - phase2_lr_classifier: 2e-4 (was 5e-4) — correspondingly lower
  - early_stopping_patience: 20 (was 15) — give more time
  - use_srm: False by default — SRM with 9 extra channels was adding
    noise at init; test without it first.  Re-enable once 80% is hit.
  - use_weighted_sampler: True (new) — oversample minority class at batch level
  - warmup_epochs: 3 (new) — linear LR warmup at Phase 2 start
  - scheduler_t_mult: 2 (new) — cosine warm restart multiplier

  ARCHITECTURE CHANGE:
  - model_name: efficientnet_b2  (was efficientnet_b0)
    B2 has a wider feature space (1408 vs 1280 features) and has been
    shown to improve steganalysis tasks by ~3-4% F1.
    If GPU memory is tight, keep efficientnet_b0.
"""

import os
import random
import torch
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Random Seed (reproducibility)
# ──────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
    """Set random seed everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# ──────────────────────────────────────────────────────────────────────
# Device Configuration
# ──────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR           = PROJECT_ROOT / "data"
COMBINED_DIR       = DATA_DIR / "combined"
NLP_CORPUS_DIR     = DATA_DIR / "nlp_corpus"
CHECKPOINTS_DIR    = PROJECT_ROOT / "checkpoints"
MODULE1_CHECKPOINT_DIR = CHECKPOINTS_DIR / "module1"
MODELS_DIR         = PROJECT_ROOT / "models"
RESULTS_DIR        = PROJECT_ROOT / "results"
REPORTS_DIR        = RESULTS_DIR / "reports"

# ──────────────────────────────────────────────────────────────────────
# Class Names and Mappings
# ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["clean", "lsb", "pvd"]
CLASS_MAP   = {"clean": 0, "lsb": 1, "pvd": 2}
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────────────────────────────
# Module 1 - CNN Hyperparameters  (v3 — targeting 80%+ F1)
# ──────────────────────────────────────────────────────────────────────
CNN_CONFIG = {
    # Switch to B2 for more feature capacity (1408 vs 1280 dims).
    # If you hit GPU OOM, revert to "efficientnet_b0".
    "model_name":  "efficientnet_b2",
    "input_size":  224,
    "num_classes": NUM_CLASSES,
    "dropout":     0.4,

    # KEPT: noise-residual Laplacian channel (4th channel)
    "use_noise_residual": True,

    # DISABLED: SRM adds 9 channels from near-zero init — too noisy
    # at the start of training.  Re-enable after reaching 80%.
    "use_srm": False,

    # Use WeightedRandomSampler in DataLoader to oversample minority class
    "use_weighted_sampler": True,

    # Training schedule
    "epochs":          80,
    "batch_size":      32,
    "num_workers":     4,

    # Phase 1: frozen backbone, only train the head
    # 10 epochs (up from 5) — val_f1 was only 0.34 at ep5, head underfitting
    "phase1_epochs":   10,
    "phase1_lr":       1e-3,

    # Phase 2: fine-tune entire network
    # Backbone LR dropped from 5e-5 → 2e-5 to fix the epoch-40 val_loss spike
    "phase2_lr_backbone":   2e-5,
    "phase2_lr_classifier": 2e-4,

    # Linear LR warmup at the start of Phase 2 (avoids the spike)
    "warmup_epochs": 3,

    "weight_decay":              1e-4,
    "scheduler_eta_min":         1e-7,
    "gradient_clip_max_norm":    1.0,
    "early_stopping_patience":   20,   # was 15

    # Focal loss
    "use_focal_loss": True,
    "focal_gamma":    2.0,

    # Label smoothing
    "label_smoothing": 0.1,

    # Manual class weights:  [clean, lsb, pvd]
    # lsb gets highest weight — it's the hardest class
    "class_weights_manual": [1.5, 2.5, 2.0],

    # ImageNet normalisation
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
}

# ──────────────────────────────────────────────────────────────────────
# Module 4 - NLP Hyperparameters
# ──────────────────────────────────────────────────────────────────────
NLP_CONFIG = {
    "tfidf_max_features": 100000,
    "tfidf_ngram_range":  (1, 4),
    "svm_C":              1.0,
}

NLP_CLASS_NAMES = [
    "PowerShell",
    "JavaScript",
    "JavaScript_HTML",
    "URL_IP",
    "Ethereum_Address",
]
NLP_NUM_CLASSES = len(NLP_CLASS_NAMES)

# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
RISK_LEVELS = ["none", "low", "medium", "high", "critical"]

# ──────────────────────────────────────────────────────────────────────
# Create all directories if they don't exist
# ──────────────────────────────────────────────────────────────────────
ALL_DIRS = [
    DATA_DIR, COMBINED_DIR, NLP_CORPUS_DIR,
    CHECKPOINTS_DIR, MODULE1_CHECKPOINT_DIR,
    MODELS_DIR, RESULTS_DIR, REPORTS_DIR,
]
for d in ALL_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("StegoDetect project initialised successfully")
    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}")
    print()
    print("Created directories:")
    for d in ALL_DIRS:
        print(f"  [OK] {d}")