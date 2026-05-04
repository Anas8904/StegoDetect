"""
StegoDetect - Project Configuration
All project constants, paths, hyperparameters, and seed settings.
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

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
COMBINED_DIR = DATA_DIR / "combined"
NLP_CORPUS_DIR = DATA_DIR / "nlp_corpus"

# Model / checkpoint directories
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODULE1_CHECKPOINT_DIR = CHECKPOINTS_DIR / "module1"
MODELS_DIR = PROJECT_ROOT / "models"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"

# ──────────────────────────────────────────────────────────────────────
# Class Names and Mappings
# ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["clean", "lsb", "pvd"]
CLASS_MAP = {"clean": 0, "lsb": 1, "pvd": 2}
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────────────────────────────
# Module 1 - CNN Hyperparameters
# ──────────────────────────────────────────────────────────────────────
CNN_CONFIG = {
    "model_name": "efficientnet_b0",
    "input_size": 224,
    "num_classes": NUM_CLASSES,
    "dropout": 0.3,
    "use_noise_residual": False,

    # Training
    "epochs": 50,
    "batch_size": 32,
    "num_workers": 4,

    # Phase 1 (frozen backbone, epochs 1-2)
    "phase1_epochs": 2,
    "phase1_lr": 1e-3,

    # Phase 2 (unfrozen, epochs 3-50)
    "phase2_lr_backbone": 1e-4,
    "phase2_lr_classifier": 1e-3,

    "weight_decay": 1e-5,
    "scheduler_eta_min": 1e-6,
    "gradient_clip_max_norm": 1.0,
    "early_stopping_patience": 10,

    # ImageNet normalization
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# ──────────────────────────────────────────────────────────────────────
# Module 4 - NLP Hyperparameters
# ──────────────────────────────────────────────────────────────────────
NLP_CONFIG = {
    "tfidf_max_features": 100000,
    "tfidf_ngram_range": (1, 4),
    "svm_C": 1.0,
}

# NLP class labels
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

# Risk level mapping
RISK_LEVELS = ["none", "low", "medium", "high", "critical"]

# ──────────────────────────────────────────────────────────────────────
# Create all directories if they don't exist
# ──────────────────────────────────────────────────────────────────────
ALL_DIRS = [
    DATA_DIR,
    COMBINED_DIR,
    NLP_CORPUS_DIR,
    CHECKPOINTS_DIR,
    MODULE1_CHECKPOINT_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
]

for d in ALL_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("StegoDetect project initialized successfully")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print()
    print("Created directories:")
    for d in ALL_DIRS:
        print(f"  [OK] {d}")
