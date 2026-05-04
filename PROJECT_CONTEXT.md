# STEGODETECT — COMPLETE PROJECT CONTEXT
> Paste this at the start of every session with your coding assistant.
> Last updated: 2026-05-04

---

## 1. PROJECT OVERVIEW

**StegoDetect** is an AI-powered Steganography Detection and Payload Analysis pipeline.
It is a capstone project that detects whether images contain hidden steganographic data,
extracts the hidden payloads, and classifies them.

### What it does (4-Module Pipeline):
1. **Module 1 — CNN Detection:** EfficientNet-B0 classifies images as `clean`, `lsb`, or `pvd`
2. **Module 2 — Statistical Validation:** Chi-square (LSB) and PPDH (PVD) tests confirm findings
3. **Module 3 — Payload Extraction:** Reverses embedding algorithms to extract hidden data
4. **Module 4 — NLP Classification:** Classifies extracted payloads (PowerShell, JavaScript, URLs, Ethereum, HTML)

### Current Status:
- [x] Dataset merging complete (originally 60,000 images, now 47,995 after test set artifact cleanup)
- [x] Module 1 CNN architecture created (EfficientNet-B0, 4M params)
- [x] Training script ready with 2-phase strategy
- [x] Gradio web UI operational (running locally at http://127.0.0.1:7860)
- [x] Module 1 training COMPLETE (Achieved 76.3% accuracy. Models saved in checkpoints/module1)
- [ ] Module 2 (statistical validator) NOT yet created
- [ ] Module 3 (payload extractor) NOT yet created
- [ ] Module 4 (NLP classifier) NOT yet created
- [ ] Full pipeline integration NOT yet done
- [ ] Evaluation suite NOT yet created

---

## 2. TECH STACK

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.1+, torchvision |
| CNN Backbone | EfficientNet-B0 via `timm` library |
| NLP (planned) | scikit-learn (TF-IDF + LinearSVC), HuggingFace Transformers (CodeBERT) |
| Image Processing | Pillow, NumPy, OpenCV |
| Web Interface | Gradio 4.0+ |
| API (planned) | FastAPI + Uvicorn |
| Statistics | SciPy |
| Visualization | Matplotlib, Seaborn |

---

## 3. DEPENDENCIES (requirements.txt)

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
transformers>=4.35.0
datasets>=2.14.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=1.0.0
tqdm>=4.65.0
gradio>=4.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
scipy>=1.11.0
joblib>=1.3.0
```

Install with: `pip install -r requirements.txt`

---

## 4. COMPLETE FILE STRUCTURE

```
Capstone/
|
|-- config.py                  # Central configuration (paths, hyperparams, seeds, class maps)
|-- requirements.txt           # Python dependencies
|-- app.py                     # Gradio web UI (3 tabs: Analyze, Batch, About)
|-- Plan.txt                   # Full implementation plan (reference document)
|-- Implementation Plan.docx   # Same plan in Word format
|-- PROJECT_CONTEXT.md         # THIS FILE - project context for AI assistants
|
|-- data/
|   |-- combined/              # THE DATASET (merged 3-class dataset)
|   |   |-- train/
|   |   |   |-- clean/         # 8,000 clean images (from both LSB & PVD sources)
|   |   |   |-- lsb/           # 12,000 LSB stego images
|   |   |   |-- pvd/           # 3,995 PVD stego images
|   |   |-- val/
|   |   |   |-- clean/         # 4,000
|   |   |   |-- lsb/           # 6,000
|   |   |   |-- pvd/           # 2,000
|   |   |-- test/
|   |   |   |-- clean/         # 4,000
|   |   |   |-- lsb/           # 6,000 (raw stego only. b64/zip variants were deleted)
|   |   |   |-- pvd/           # 2,000
|   |   |-- class_map.json     # {"clean": 0, "lsb": 1, "pvd": 2}
|   |   |-- merge_summary.json # Statistics from the dataset merge
|   |   |-- merge_errors.log   # Error log (should be empty)
|   |-- nlp_corpus/            # (empty - will hold NLP training data)
|
|-- data_prep/
|   |-- __init__.py
|   |-- merge_datasets.py      # Script that merged LSB + PVD datasets into data/combined/
|   |-- verify_dataset.py      # Dataset integrity checker (512x512, PNG, corruption)
|
|-- modules/
|   |-- __init__.py
|   |-- dataset.py             # PyTorch StegoDataset + DataLoader + transforms
|   |-- module1_cnn.py         # EfficientNet-B0 StegoClassifier model
|
|-- training/
|   |-- __init__.py
|   |-- train_module1.py       # Full training script for Module 1 CNN
|
|-- checkpoints/               # (will hold trained model weights)
|   |-- module1/               # best_model.pth, last_model.pth, etc.
|
|-- models/                    # (will hold NLP models)
|   |-- codebert/              # CodeBERT weights
|
|-- results/                   # (will hold evaluation outputs)
|   |-- reports/               # JSON reports from pipeline analysis
```

---

## 5. DETAILED FILE DESCRIPTIONS

### config.py
Central configuration file imported by ALL other modules. Contains:
- **SEED = 42** — set everywhere (torch, numpy, random)
- **DEVICE** — auto-detects CUDA/CPU
- **PROJECT_ROOT** — auto-resolves from file location
- **All directory paths** — DATA_DIR, COMBINED_DIR, CHECKPOINTS_DIR, etc.
- **CLASS_NAMES** = ["clean", "lsb", "pvd"], CLASS_MAP = {clean:0, lsb:1, pvd:2}
- **CNN_CONFIG** — all hyperparameters for Module 1:
  - EfficientNet-B0, input 224x224, dropout 0.3
  - 50 epochs, batch_size 32, 4 workers
  - Phase 1 (epochs 1-10): frozen backbone, lr=1e-3
  - Phase 2 (epochs 11-50): unfrozen, backbone lr=1e-4, head lr=1e-3
  - AdamW, CosineAnnealing, early stopping patience=10
  - ImageNet normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- **NLP_CONFIG** — hyperparameters for Module 4 (TF-IDF, CodeBERT settings)
- Auto-creates all directories on import

### modules/dataset.py
PyTorch data pipeline for Module 1 training:
- **StegoDataset(Dataset)** — walks `data/combined/{split}/{class}/` to build (path, label) pairs
  - Computes class weights for imbalanced dataset: `total / (num_classes * count_per_class)`
  - Optional noise-residual 4th channel via Laplacian filter
- **get_transforms(split)** — careful augmentations that preserve stego signatures:
  - Train: Resize(224) + RandomHFlip + RandomVFlip + ToTensor + Normalize
  - Val/Test: Resize(224) + ToTensor + Normalize
  - NO ColorJitter, NO RandomRotation, NO RandomCrop (these destroy stego patterns)
- **get_dataloaders()** — returns train/val/test loaders + class_weights

### modules/module1_cnn.py
The CNN model for steganography detection:
- **StegoClassifier(nn.Module)** — EfficientNet-B0 backbone + classification head
  - Backbone: `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)`
  - Feature dim: 1280
  - Head: Dropout(0.3) → Linear(1280, 3)
  - Optional 4-channel input (noise-residual): modifies conv_stem to accept 4 channels
  - `forward(x)` → logits (batch, 3)
  - `predict_with_confidence(x)` → (predicted_class, confidence, probabilities)
- **Total params:** 4,011,391 (15.3 MB)
- Utility functions: `freeze_backbone_layers()`, `unfreeze_all()`, `get_model_summary()`, `load_pretrained_stego()`

### training/train_module1.py
Complete training script with:
- **Two-Phase Training:**
  - Phase 1 (epochs 1-10): Freeze backbone, train only classifier head, lr=1e-3
  - Phase 2 (epochs 11-50): Unfreeze all, differential lr (backbone=1e-4, head=1e-3)
- **Weighted CrossEntropyLoss** (handles class imbalance)
- **AdamW optimizer** with weight_decay=1e-5
- **CosineAnnealingLR scheduler**
- **Gradient clipping** max_norm=1.0
- **Early stopping** patience=10, based on val F1 macro
- **Checkpointing:** best_model.pth (by F1), last_model.pth (every epoch), periodic every 10 epochs
- **Metrics:** per-class precision/recall/F1, confusion matrix (saved as PNG), ROC curves
- **Final evaluation** on test set: accuracy, classification report, clean FPR (<2% target)
- **Resume training** with `--resume` flag
- **CLI:** `python training/train_module1.py --epochs 50 --batch_size 32 --device cuda`

### app.py
Gradio web interface with 3 tabs:
- **Tab 1 "Analyze Image":** Upload single image → detection result, risk level, confidence, optional technical details, JSON report download
- **Tab 2 "Batch Analysis":** Upload multiple images → summary table (Filename, Technique, Confidence, Risk Level)
- **Tab 3 "About":** Project info, steganography explanation, dataset info, tech stack
- Loads `checkpoints/module1/best_model.pth` on startup
- Shows warning if models not trained yet

### data_prep/merge_datasets.py
Already executed — merged two raw datasets into `data/combined/`:
- LSB dataset (was in `archive/train/train/clean`, `archive/train/train/stego`, etc.)
- PVD dataset (was in `Stego-pvd-dataset/train/cleanTrain`, etc.)
- Clean images from both sources → `combined/train/clean/` (lsb_ and pvd_ prefixes)
- LSB stego → `combined/train/lsb/` (b64_ and zip_ prefixes for variants)
- PVD stego → `combined/train/pvd/`
- Raw datasets have been DELETED after successful merge

### data_prep/verify_dataset.py
Verifies dataset integrity:
- Checks all images are 512x512 PNG
- Checks all images can be opened
- Prints pass/fail summary
- Usage: `python data_prep/verify_dataset.py --dir ./data/combined/`

---

## 6. DATASET DETAILS

All images are **512x512 pixels, PNG format**.

### Class Distribution:
| Split | Clean | LSB | PVD | Total |
|-------|-------|-----|-----|-------|
| Train | 8,000 | 12,000 | 3,995 | 23,995 |
| Val | 4,000 | 6,000 | 2,000 | 12,000 |
| Test | 4,000 | 6,000 | 2,000 | 12,000 |
| **Total** | **16,000** | **24,000** | **7,995** | **47,995** |

### Class Weights (train split):
- clean: 1.00
- lsb: 0.67
- pvd: 2.00

### LSB Variants in Test Set:
The test/lsb/ folder originally contained 18,000 images, but the `b64` and `zip` payload variants were DELETED to clean the dataset artifacts. It now contains exactly 6,000 images:
- 6,000 raw stego (no prefix)
- ~~6,000 base64-encoded payload (`b64_` prefix)~~ [DELETED]
- ~~6,000 zip-compressed payload (`zip_` prefix)~~ [DELETED]

Train and Val LSB folders only have raw stego images (no b64/zip variants).

---

## 7. MODULES STILL TO BE BUILT

### Module 2 — Statistical Validator (modules/module2_validator.py)
- `chi_square_lsb_test(image_path)` — detect LSB via pixel histogram analysis
- `ppdh_pvd_test(image_path)` — detect PVD via pixel-pair difference histogram
- `validate_and_route(image_path, cnn_prediction, cnn_confidence)` — combine CNN + stats
- See Plan.txt PHASE 5 for full specification

### Module 3 — Payload Extractor (modules/module3_extractor.py)
- `extract_lsb(image_path)` — extract LSB-hidden bits, pack into bytes
- `extract_pvd(image_path)` — extract PVD-hidden data using quantization table
- `detect_obfuscation(raw_bytes)` — detect base64/zip/none
- `deobfuscate(raw_bytes)` — reverse obfuscation layers (recursive up to 3 deep)
- `assess_extraction_quality(raw_bytes)` — quality: good/partial/failed
- See Plan.txt PHASE 6 for full specification

### Module 4 — NLP Payload Classifier (modules/module4_nlp.py)
- TF-IDF + LinearSVC as primary classifier
- Optional CodeBERT for obfuscated payloads
- 5 classes: PowerShell, JavaScript, JavaScript_HTML, URL_IP, Ethereum_Address
- See Plan.txt PHASE 8 & 9 for full specification

### Pipeline Integration (pipeline.py)
- StegoDetectPipeline class chaining all 4 modules
- See Plan.txt PHASE 10 for full specification

### Evaluation Suite (evaluate.py)
- Module-level and end-to-end evaluation
- See Plan.txt PHASE 11 for full specification

### NLP Corpus Builder (data_prep/build_nlp_corpus.py)
- Extracts payloads from stego images to build NLP training data
- See Plan.txt PHASE 7 for full specification

---

## 8. HOW TO RUN

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize project (creates all directories)
python config.py

# Verify dataset integrity
python data_prep/verify_dataset.py --dir ./data/combined/

# Train Module 1 CNN (use --device cuda if GPU available)
python training/train_module1.py --epochs 50 --batch_size 32 --device cuda

# Resume interrupted training
python training/train_module1.py --resume --device cuda

# Launch web interface (after training)
python app.py
# Opens at http://localhost:7860
```

---

## 9. IMPORTANT NOTES

### Windows Compatibility
- Console encoding is cp1252 — avoid Unicode characters (checkmarks, arrows) in print statements
- Use `[OK]`, `[FAIL]`, `[WARN]` etc. instead of Unicode symbols
- Paths use backslashes but pathlib handles this automatically

### Dataset Augmentation Warning
Do NOT use these transforms on stego images (they destroy embedded data patterns):
- ColorJitter
- RandomRotation
- RandomCrop
- Any pixel-value modifying transform

Only safe augmentations: RandomHorizontalFlip, RandomVerticalFlip (spatial flips preserve LSB/PVD patterns)

### Class Imbalance
PVD class is underrepresented (~17% of train). Handled via:
- Weighted CrossEntropyLoss (PVD weight = 2.0)
- Class weights computed automatically by StegoDataset

### Common Issues
- CUDA OOM: reduce batch_size to 16, add `torch.cuda.empty_cache()` after each epoch
- NaN loss: reduce lr by 10x, add gradient clipping (already set to 1.0)
- See Plan.txt "COMMON ERRORS AND FIXES" section for more

---

## 10. REFERENCE DOCUMENT

The full implementation plan with exact specifications for ALL modules is in `Plan.txt`.
Refer to it for detailed pseudocode, function signatures, and expected behavior.
Key sections:
- PHASE 1: Dataset Merging (DONE)
- PHASE 2: Data Pipeline / DataLoader (DONE)
- PHASE 3: Module 1 CNN Architecture (DONE)
- PHASE 4: Training Module 1 (DONE - trained with 76.3% accuracy, best model saved)
- PHASE 5: Module 2 Statistical Validator (TODO)
- PHASE 6: Module 3 Payload Extractor (TODO)
- PHASE 7: NLP Corpus Builder (TODO)
- PHASE 8-9: Module 4 NLP Classifier (TODO)
- PHASE 10: Pipeline Integration (TODO)
- PHASE 11: Evaluation Suite (TODO)
- PHASE 12: Gradio Web Interface (DONE - CNN part ready and operational locally)
- PHASE 13: Run Order & Integration Test (TODO)
