"""
StegoDetect - Project Configuration
All project constants, paths, hyperparameters, and seed settings.

CHANGES FROM v1:
  - use_noise_residual: True  (was False) -- critical for LSB detection
  - phase1_epochs: 5          (was 2)     -- longer warm-up before unfreeze
  - epochs: 75                (was 50)    -- model stopped improving at ep28
  - phase2_lr_backbone: 5e-5  (was 1e-4)  -- slower backbone fine-tuning
  - phase2_lr_classifier: 5e-4 (was 1e-3) -- slower head fine-tuning
  - weight_decay: 1e-4        (was 1e-5)  -- better regularisation
  - early_stopping_patience: 15 (was 10)  -- give model more time
  - dropout: 0.4              (was 0.3)
  - label_smoothing: 0.1      (new)
  - focal_gamma: 2.0          (new)       -- focal loss exponent
  - use_focal_loss: True      (new)
  - class_weights_manual: [1.5, 2.5, 2.0] (new)  -- overrides auto-compute
    * auto-compute gave LSB weight=0.67, which was HURTING recall
    * we now give LSB more weight to force the model to attend to it
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
# Module 1 - CNN Hyperparameters  (v2 — tuned for 85 %+ F1)
# ──────────────────────────────────────────────────────────────────────
CNN_CONFIG = {
    "model_name":  "efficientnet_b0",
    "input_size":  224,
    "num_classes": NUM_CLASSES,
    "dropout":     0.4,               # v1=0.3  → slightly stronger regularisation

    # KEY CHANGE: noise-residual 4th channel
    # A Laplacian-filtered version of the image is added as channel 4.
    # LSB steganography embeds in the least-significant bits, which shows
    # up as subtle high-frequency noise — exactly what this channel captures.
    "use_noise_residual": True,       # v1=False

    # Training schedule
    "epochs":          75,            # v1=50  (model was at ep28/50, could improve)
    "batch_size":      32,
    "num_workers":     4,

    # Phase 1: frozen backbone, only train the head
    "phase1_epochs":   5,             # v1=2   (more warm-up before unfreezing)
    "phase1_lr":       1e-3,

    # Phase 2: fine-tune entire network with differential LR
    "phase2_lr_backbone":   5e-5,     # v1=1e-4  (slower = more stable)
    "phase2_lr_classifier": 5e-4,     # v1=1e-3  (slower = more stable)

    "weight_decay":              1e-4, # v1=1e-5
    "scheduler_eta_min":         1e-7, # v1=1e-6
    "gradient_clip_max_norm":    1.0,
    "early_stopping_patience":   15,   # v1=10

    # Focal loss (new)
    # Standard cross-entropy treats every misclassification equally.
    # Focal loss down-weights easy correct predictions and concentrates
    # the gradient on the hard misclassified LSB examples.
    "use_focal_loss": True,            # v1: plain CE
    "focal_gamma":    2.0,             # standard value from Lin et al. 2017

    # Label smoothing (new)
    # Prevents over-confident softmax, improves calibration.
    "label_smoothing": 0.1,            # v1: no smoothing

    # Manual class weights (new)
    # v1 auto-computed:  clean=1.00, lsb=0.67, pvd=2.00
    # The 0.67 weight for LSB was actively REDUCING the loss for missed
    # LSB images, contributing to the 70% recall.  We flip this:
    # lsb gets the highest weight so misses are penalised more.
    "class_weights_manual": [1.5, 2.5, 2.0],  # [clean, lsb, pvd]

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