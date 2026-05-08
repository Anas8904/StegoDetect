# STEGODETECT — COMPLETE PROJECT CONTEXT
> Paste this at the start of every session with your coding assistant.
> Last updated: 2026-05-08

---

## 1. PROJECT OVERVIEW

**StegoDetect** is an AI-powered Steganography Detection and Payload Analysis pipeline.
It is a capstone project that detects whether images contain hidden steganographic data, extracts the hidden payloads, and classifies them.

### What it does (4-Module Pipeline):
1.  **Module 1 — CNN Detection:** EfficientNet-B4 classifies images as `clean`, `lsb`, or `pvd`.
2.  **Module 2 — Statistical Validation:** Chi-square (LSB) and PPDH (PVD) tests confirm findings.
3.  **Module 3 — Payload Extraction:** Reverses embedding algorithms to extract hidden data (LSB/PVD).
4.  **Module 4 — NLP Classification:** 3-stage classifier (Regex, TF-IDF+SVM, CodeBERT) for payload analysis.

### Current Status:
- [x] **Dataset Merge:** 59,995 images (512x512 PNG) fully prepared and verified.
- [x] **Module 1 (CNN):** Upgraded to **EfficientNet-B4**. Training script supports AMP, Warmup, and Label Smoothing.
- [x] **Module 2 (Stats):** Fully implemented (Chi-square for LSB, PPDH for PVD).
- [x] **Module 3 (Extractor):** Fully implemented with recursive deobfuscation (B64/ZIP).
- [x] **Module 4 (NLP):** Fully implemented 3-stage pipeline (Regex, TF-IDF+SVM, CodeBERT).
- [x] **Full Pipeline Integration:** End-to-end inference implemented in `pipeline.py`.
- [x] **Premium Web UI:** Upgraded Gradio interface with modern design and full module integration.
- [x] **Evaluation Suite:** Module-level evaluation ready; end-to-end report generation in pipeline.

---

## 2. RECENT UPGRADES (May 2026)

### EfficientNet-B4 Architecture
- **Higher Resolution:** Upgraded from B0 to B4, increasing input resolution to 380x380 for better detail capture.
- **Enhanced Capacity:** 19M parameters (vs 5.3M) allow for learning more complex steganographic patterns.
- **Advanced Training:** Integrated Mixed Precision (AMP), Linear LR Warmup, and Label Smoothing (0.1) for superior convergence.

### Module 4 — NLP Classification
- **Stage 1 (Regex):** Instant classification for unambiguous payloads (Ethereum addresses, URLs/IPs).
- **Stage 2 (TF-IDF + SVM):** Robust character-level analysis for code classification (JS, HTML, PowerShell).
- **Stage 3 (CodeBERT):** Context-aware transformer model for handling obfuscated or complex payloads.

### Premium Web Experience
- **Modern UI:** Redesigned Gradio interface with Inter typography, sleek dark/gradient themes, and interactive result displays.
- **Detailed Insights:** Real-time risk level assessment (Critical to None) and integrated extraction previews.
- **JSON Reports:** Automatic generation of comprehensive forensic reports.

---

## 3. TECH STACK

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.1+, torchvision, `timm`, `transformers` |
| Image Processing | OpenCV, Pillow, NumPy |
| Statistics | SciPy |
| Web Interface | Gradio 4.0+ (Premium UI) |
| NLP | scikit-learn (TF-IDF + LinearSVC), CodeBERT |
| Data Handling | Pandas, Joblib |

---

## 4. COMPLETE FILE STRUCTURE

```
stego2/
|-- config.py                  # Central configuration (paths, hyperparams, seeds, class maps)
|-- requirements.txt           # Python dependencies
|-- app.py                     # Premium Gradio web UI (full 4-module integration)
|-- pipeline.py                # Standalone end-to-end inference pipeline
|-- PROJECT_CONTEXT.md         # THIS FILE - project context for AI assistants
|
|-- data/
|   |-- combined/              # THE DATASET (merged 3-class dataset)
|   |   |-- train/, val/, test/
|   |-- nlp_corpus/            # Extracted payload data for Module 4 training
|
|-- data_prep/
|   |-- merge_datasets.py      # Merges LSB + PVD datasets
|   |-- verify_dataset.py      # Dataset integrity checker
|   |-- build_nlp_corpus.py    # Extracts payloads to build NLP training data
|
|-- modules/
|   |-- dataset.py             # PyTorch StegoDataset (SRM residuals, flips only)
|   |-- module1_cnn.py         # EfficientNet-B4 StegoClassifier
|   |-- module2_validator.py   # Statistical validation (Chi-square, PPDH)
|   |-- module3_extractor.py   # Payload extraction and deobfuscation
|   |-- module4_nlp.py         # 3-stage NLP payload classifier
|
|-- training/
|   |-- train_module1.py       # Upgraded B4 training script (AMP, Warmup, Smoothing)
|   |-- train_module4.py       # Training script for TF-IDF/SVM and CodeBERT
|
|-- checkpoints/               # Trained model weights (module1, tfidf_svm, codebert)
|-- results/                   # Forensic reports and evaluation metrics
```

---

## 5. DETAILED FILE DESCRIPTIONS

### modules/module1_cnn.py
- **EfficientNet-B4 Backbone:** Upgraded for higher accuracy; 380x380 input resolution.
- **Enhanced Head:** Multi-layer classification head with BatchNorm and high dropout (0.4).

### modules/module4_nlp.py
- **Stage 1 (Regex):** Fast-path for ETH (0x...) and URL/IP patterns.
- **Stage 2 (TF-IDF + SVM):** N-gram based SVC for robust language detection.
- **Stage 3 (CodeBERT):** Optional transformer stage for deep code analysis.

### pipeline.py
- **End-to-End Inference:** Chains all 4 modules: CNN -> Stats -> Extract -> NLP.
- **Risk Assessment:** Categorizes detections into Critical, High, Medium, Low, or None.

### app.py
- **Premium Interface:** High-fidelity Gradio UI with forensic report downloads and technical detail toggles.

---

## 6. HOW TO RUN

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train Module 1 (B4)
python training/train_module1.py --device cuda --batch_size 16

# 3. Build NLP Corpus & Train Module 4
python data_prep/build_nlp_corpus.py
python training/train_module4.py --skip_codebert

# 4. Run Web UI
python app.py
```

---

## 7. IMPORTANT NOTES

- **Signal Integrity:** NEVER use destructive augmentations (Blur, ColorJitter, Rotation) or JPEG compression. Use spatial flips only.
- **B4 Performance:** EfficientNet-B4 requires more VRAM than B0; batch size 16 is recommended for 8-12GB GPUs.
- **NLP Heuristics:** The system uses keyword fallbacks for NLP classification if the TF-IDF model is not yet trained.
