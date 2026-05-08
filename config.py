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
CODEBERT_DIR = MODELS_DIR / "codebert"

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
# Module 1 - CNN Hyperparameters (EfficientNet-B4)
# ──────────────────────────────────────────────────────────────────────
CNN_CONFIG = {
    "model_name": "efficientnet_b4",
    "input_size": 380,              # B4 optimal input size
    "num_classes": NUM_CLASSES,
    "dropout": 0.4,                 # Higher dropout for B4 (larger model)
    "use_noise_residual": False,
    "label_smoothing": 0.1,         # Prevents overconfident predictions

    # Training
    "epochs": 50,
    "batch_size": 16,               # Reduced for B4 VRAM requirements
    "num_workers": 4,
    "use_amp": True,                # Mixed precision training for speed

    # Phase 1 (frozen backbone, epochs 1-5)
    "phase1_epochs": 5,
    "phase1_lr": 5e-4,

    # Phase 2 (unfrozen, epochs 6-50)
    "phase2_lr_backbone": 5e-5,     # Very low LR for pretrained B4 backbone
    "phase2_lr_classifier": 5e-4,

    # Warmup
    "warmup_epochs": 3,             # Linear LR warmup for stability
    "warmup_start_factor": 0.01,    # Start at 1% of target LR

    "weight_decay": 1e-4,
    "scheduler_eta_min": 1e-7,
    "gradient_clip_max_norm": 1.0,
    "early_stopping_patience": 12,

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
    "codebert_model_name": "microsoft/codebert-base",
    "codebert_max_length": 512,
    "codebert_batch_size": 8,
    "codebert_epochs": 5,
    "codebert_lr": 2e-5,
}

# NLP class labels
NLP_CLASS_NAMES = [
    "JavaScript",
    "JavaScript_HTML",
    "PowerShell",
    "URL_IP",
    "Ethereum_Address",
]
NLP_LABEL_MAP = {i: name for i, name in enumerate(NLP_CLASS_NAMES)}
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
    CODEBERT_DIR,
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
    print(f"CNN Model: {CNN_CONFIG['model_name']} (input: {CNN_CONFIG['input_size']}x{CNN_CONFIG['input_size']})")
    print()
    print("Created directories:")
    for d in ALL_DIRS:
        print(f"  [OK] {d}")
