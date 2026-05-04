MASTER PROJECT CONTEXT BLOCK
(Paste this at the start of EVERY session with your coding assistant)
PROJECT: StegoDetect — AI-Powered Steganography Detection and Payload Analysis

WHAT IT DOES: A 4-module end-to-end pipeline that:
1. Detects whether an image contains hidden steganographic data (Module 1 - CNN)
2. Validates the detection and routes to the correct extractor (Module 2 - Statistics)
3. Extracts the hidden payload from the image (Module 3 - Algorithm)
4. Classifies the payload type using NLP (Module 4 - NLP)

TECH STACK:
- Python 3.10+
- PyTorch 2.1+ with torchvision
- EfficientNet-B0 (pretrained ImageNet weights via timm library)
- scikit-learn (TF-IDF, LinearSVC)
- Hugging Face Transformers (CodeBERT)
- Pillow, NumPy, OpenCV
- Gradio (web interface)
- FastAPI (REST API)

DATASET STRUCTURE:
Dataset A (LSB Steganography) - located at ./data/lsb_dataset/
    train/
        clean/      (original unmodified images)
        stego/      (LSB-embedded images with raw payloads)
        stego_b64/  (LSB-embedded images with base64-encoded payloads)
        stego_zip/  (LSB-embedded images with zip-compressed payloads)
    test/
        clean/
        stego/
        stego_b64/
        stego_zip/
    val/
        clean/
        stego/
        stego_b64/
        stego_zip/

Dataset B (PVD Steganography) - located at ./data/pvd_dataset/
    train/
        clean/
        stego/
    test/
        clean/
        stego/
    val/
        clean/
        stego/

ALL IMAGES: 512x512 pixels, PNG format

MODULE 1 CLASSES: 3-class classification
    Class 0 = clean
    Class 1 = lsb (includes stego, stego_b64, stego_zip from LSB dataset)
    Class 2 = pvd (stego from PVD dataset)

MODULE 4 CLASSES: 5-class payload classification
    Class 0 = JavaScript
    Class 1 = JavaScript_in_HTML
    Class 2 = PowerShell
    Class 3 = URL_IP
    Class 4 = Ethereum_Address

IMPORTANT RULES:
- Never use JPEG compression anywhere in the pipeline (destroys stego signatures)
- Never use color jitter, rotation, or brightness augmentation (corrupts stego statistics)
- Only safe augmentations: horizontal flip, vertical flip
- Always save images as PNG
- Input images must be exactly 512x512 before any processing

PHASE 0 — PROJECT SETUP
Prompt for coding assistant:
Using the project context above, set up the complete StegoDetect project structure.

Create the following:

1. DIRECTORY STRUCTURE:
Create this exact folder structure:
StegoDetect/
??? data/
?   ??? lsb_dataset/          (empty - user will populate)
?   ??? pvd_dataset/          (empty - user will populate)
?   ??? combined/             (will be created by dataset merger script)
?       ??? train/
?       ?   ??? clean/
?       ?   ??? lsb/
?       ?   ??? pvd/
?       ??? val/
?       ?   ??? clean/
?       ?   ??? lsb/
?       ?   ??? pvd/
?       ??? test/
?           ??? clean/
?           ??? lsb/
?           ??? pvd/
??? modules/
?   ??? __init__.py
?   ??? module1_cnn.py
?   ??? module2_validator.py
?   ??? module3_extractor.py
?   ??? module4_nlp.py
??? training/
?   ??? train_module1.py
?   ??? train_module4.py
??? data_prep/
?   ??? merge_datasets.py
?   ??? build_nlp_corpus.py
?   ??? verify_dataset.py
??? pipeline.py
??? evaluate.py
??? app.py
??? api.py
??? requirements.txt
??? config.py

2. requirements.txt with these exact packages:
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

3. config.py with all project constants:
- All directory paths
- Model hyperparameters
- Class names and mappings
- Device configuration (CUDA if available, else CPU)
- Random seed = 42

Set random seed everywhere: torch, numpy, random module.
Create all directories if they don't exist.
Add a main block that prints "StegoDetect project initialized successfully" 
and lists all created directories.

PHASE 1 — DATASET MERGING
Prompt for coding assistant:
Using the StegoDetect project context, create data_prep/merge_datasets.py

This script combines two separate datasets into one unified dataset for 3-class 
classification (clean, lsb, pvd).

DATASET LOCATIONS:
- LSB dataset: ./data/lsb_dataset/
- PVD dataset: ./data/pvd_dataset/
- Output: ./data/combined/

MERGING RULES:

For the CLEAN class (class 0):
- Source 1: ./data/lsb_dataset/train/clean/ ? copy to ./data/combined/train/clean/
- Source 2: ./data/pvd_dataset/train/clean/ ? copy to ./data/combined/train/clean/
- Handle filename conflicts: if both datasets have "image_001.png", rename PVD 
  images with prefix "pvd_" (e.g., "pvd_image_001.png")
- Same logic for test/ and val/ splits

For the LSB class (class 1):
- Source 1: ./data/lsb_dataset/train/stego/ ? copy to ./data/combined/train/lsb/
- Source 2: ./data/lsb_dataset/train/stego_b64/ ? copy to ./data/combined/train/lsb/
  with prefix "b64_"
- Source 3: ./data/lsb_dataset/train/stego_zip/ ? copy to ./data/combined/train/lsb/
  with prefix "zip_"
- Same logic for test/ and val/ splits

For the PVD class (class 2):
- Source: ./data/pvd_dataset/train/stego/ ? copy to ./data/combined/train/pvd/
- Same logic for test/ and val/ splits

ALSO CREATE: ./data/combined/class_map.json with content:
{
    "clean": 0,
    "lsb": 1,
    "pvd": 2
}

IMPORTANT IMPLEMENTATION DETAILS:
- Use shutil.copy2 to preserve file metadata
- DO NOT move files - always copy so original datasets remain intact
- Show a progress bar using tqdm
- After merging, print a summary table showing:
  Split | Clean Count | LSB Count | PVD Count | Total
  train | ...         | ...        | ...        | ...
  val   | ...         | ...        | ...        | ...
  test  | ...         | ...        | ...        | ...
- Verify all copied images can be opened with Pillow (corruption check)
- Log any failed copies to ./data/combined/merge_errors.log
- Save merge statistics to ./data/combined/merge_summary.json

Also create data_prep/verify_dataset.py that:
- Checks all images are exactly 512x512
- Checks all images are PNG
- Checks all images can be opened
- Reports any issues found
- Prints pass/fail summary

PHASE 2 — DATA PIPELINE (PYTORCH DATALOADER)
Prompt for coding assistant:
Using the StegoDetect project context, create modules/dataset.py

This file contains the PyTorch Dataset and DataLoader for Module 1 training.

CLASS 1: StegoDataset(torch.utils.data.Dataset)

__init__(self, root_dir, split='train', transform=None, use_noise_residual=False):
    - root_dir: path to ./data/combined/
    - split: 'train', 'val', or 'test'
    - transform: torchvision transforms
    - use_noise_residual: if True, add 4th channel (explained below)
    
    Build a list of (image_path, label) tuples by walking:
    root_dir/split/clean/ ? label 0
    root_dir/split/lsb/   ? label 1
    root_dir/split/pvd/   ? label 2
    
    Store class weights for weighted loss:
    Compute as: total_samples / (num_classes * count_per_class)

__len__(self): return total number of samples

__getitem__(self, idx):
    1. Load image with Pillow, convert to RGB
    2. Apply transform
    3. If use_noise_residual is True:
       - Convert image tensor to numpy
       - Apply Laplacian filter using cv2.Laplacian(img_gray, cv2.CV_64F)
       - Normalize residual to [0,1]
       - Concatenate as 4th channel (shape becomes 4 x H x W)
    4. Return (image_tensor, label)

CLASS 2: PayloadDataset(torch.utils.data.Dataset)
For Module 4 NLP training.

__init__(self, texts, labels, tokenizer, max_length=512):
    texts: list of extracted payload strings
    labels: list of integer class labels (0-4)
    tokenizer: HuggingFace tokenizer

__getitem__(self, idx):
    Tokenize texts[idx] with:
    - truncation=True
    - max_length=self.max_length
    - padding='max_length'
    - return_tensors='pt'
    Return input_ids, attention_mask, label

FUNCTION: get_transforms(split)
For 'train': Compose([
    Resize((224, 224)),           # EfficientNet input size
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
For 'val' and 'test': Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
NOTE: Do NOT use ColorJitter, RandomRotation, RandomCrop, or any transform 
that modifies pixel values or spatial arrangement — these destroy stego signatures.

FUNCTION: get_dataloaders(config)
Returns train_loader, val_loader, test_loader
- train: shuffle=True, batch_size=32, num_workers=4, pin_memory=True
- val/test: shuffle=False, batch_size=64, num_workers=4

Print dataset statistics when this module is run directly.

PHASE 3 — MODULE 1 (CNN STEGANALYSIS)
Prompt for coding assistant:
Using the StegoDetect project context, create modules/module1_cnn.py

This is the core steganalysis classifier using EfficientNet-B0.

CLASS: StegoClassifier(nn.Module)

__init__(self, num_classes=3, use_noise_residual=False, dropout=0.3):
    
    Load EfficientNet-B0 from timm library:
    self.backbone = timm.create_model(
        'efficientnet_b0', 
        pretrained=True,
        num_classes=0,        # Remove original classification head
        global_pool='avg'
    )
    
    Get the feature dimension from the backbone:
    feature_dim = self.backbone.num_features  # Should be 1280 for B0
    
    If use_noise_residual is True:
        Modify the first convolution layer to accept 4 channels instead of 3:
        original_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(4, original_conv.out_channels,
                            kernel_size=original_conv.kernel_size,
                            stride=original_conv.stride,
                            padding=original_conv.padding,
                            bias=False)
        Copy original weights for first 3 channels:
        new_conv.weight.data[:, :3, :, :] = original_conv.weight.data
        Initialize 4th channel weights to near-zero (0.01 * random normal)
        Replace: self.backbone.conv_stem = new_conv
    
    Classification head:
    self.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(feature_dim, num_classes)
    )

forward(self, x):
    features = self.backbone(x)
    return self.classifier(features)

predict_with_confidence(self, x):
    self.eval()
    with torch.no_grad():
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    return predicted_class, confidence, probs

FUNCTION: freeze_backbone_layers(model, num_layers_to_freeze)
    Freeze the first num_layers_to_freeze blocks of the backbone.
    This allows fine-tuning only the later layers initially.

FUNCTION: get_model_summary(model)
    Print total parameters, trainable parameters, model size in MB.

FUNCTION: load_pretrained_stego(checkpoint_path, device)
    Load a saved StegoClassifier checkpoint.
    Return the model in eval mode.

When run directly, create a model instance and run a test forward pass 
with a random batch of shape (4, 3, 224, 224) and print output shapes 
and predicted classes.

PHASE 4 — TRAINING MODULE 1
Prompt for coding assistant:
Using the StegoDetect project context, create training/train_module1.py

This script handles the complete training loop for Module 1 (EfficientNet-B0).

TRAINING CONFIGURATION:
- Epochs: 50
- Batch size: 32
- Learning rate: 1e-4
- Weight decay: 1e-5
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR with T_max=50, eta_min=1e-6
- Early stopping: patience=10 (stop if val loss doesn't improve for 10 epochs)
- Loss: CrossEntropyLoss with class weights from dataset
- Gradient clipping: max_norm=1.0
- Save best model based on validation F1-score (not just accuracy)

TWO-PHASE TRAINING STRATEGY:
Phase 1 (Epochs 1-10): 
    - Freeze all backbone layers
    - Only train the classification head
    - Higher learning rate: 1e-3
    - This prevents destroying pretrained features early in training

Phase 2 (Epochs 11-50):
    - Unfreeze all layers
    - Lower learning rate: 1e-4 for backbone, 1e-3 for classifier head
    - Use different learning rates via parameter groups:
      optimizer = AdamW([
          {'params': model.backbone.parameters(), 'lr': 1e-4},
          {'params': model.classifier.parameters(), 'lr': 1e-3}
      ], weight_decay=1e-5)

TRAINING LOOP:
For each epoch:
    1. Train phase:
       - Forward pass
       - Compute weighted cross-entropy loss
       - Backward pass with gradient clipping
       - Update weights
       - Track: loss, accuracy per class, overall accuracy
    
    2. Validation phase:
       - No gradient computation
       - Compute: loss, accuracy, precision, recall, F1 per class
       - Compute confusion matrix
       - Track best F1 score
    
    3. Logging:
       - Print epoch summary with all metrics
       - Save metrics to ./checkpoints/training_log.csv
       - Save confusion matrix plot every 5 epochs

CHECKPOINTING:
Save to ./checkpoints/module1/:
- best_model.pth: saved whenever val F1 improves
  Contains: model_state_dict, optimizer_state_dict, epoch, val_f1, config
- last_model.pth: saved every epoch (for resuming)
- checkpoint_epoch_N.pth: saved every 10 epochs

RESUME TRAINING:
Add --resume flag that loads last_model.pth and continues training.

FINAL EVALUATION:
After training completes:
- Load best_model.pth
- Run full evaluation on test set
- Print and save:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Confusion matrix (save as PNG)
  - Classification report
  - False positive rate on clean images (critical metric - must be < 2%)
  - ROC curves for each class

COMMAND LINE INTERFACE:
python train_module1.py --epochs 50 --batch_size 32 --lr 1e-4 
                        --use_noise_residual --resume --device cuda

PHASE 5 — MODULE 2 (STATISTICAL VALIDATOR)
Prompt for coding assistant:
Using the StegoDetect project context, create modules/module2_validator.py

This module performs statistical validation of Module 1's predictions 
using two classical steganalysis tests.

FUNCTION 1: chi_square_lsb_test(image_path)

PURPOSE: Detect LSB steganography via pixel histogram analysis.

STEPS:
1. Load image as numpy array using Pillow
2. For each channel (R, G, B) separately:
   a. Extract all pixel values for that channel as 1D array (shape: 512*512,)
   b. Count frequency of each value 0-255 using np.bincount(channel, minlength=256)
   c. Group into 128 pairs: (0,1), (2,3), (4,5)... (254,255)
   d. For each pair (count_even, count_odd):
      expected = (count_even + count_odd) / 2
      if expected == 0: skip this pair (avoid division by zero)
      chi2_contribution = ((count_even - expected)**2 / expected) + 
                          ((count_odd - expected)**2 / expected)
   e. Sum all contributions ? total chi-square value for this channel
   f. Calculate p-value using scipy.stats.chi2.sf(chi2_value, df=127)
      (127 degrees of freedom = 128 pairs - 1)

3. Average p-values across all 3 channels

4. Return:
{
    "lsb_detected": bool (True if avg_p_value < 0.05),
    "p_value": float,
    "chi2_value": float,
    "confidence": float (1 - p_value, capped at 0.99),
    "per_channel": {"R": p_val_r, "G": p_val_g, "B": p_val_b}
}

FUNCTION 2: ppdh_pvd_test(image_path)

PURPOSE: Detect PVD steganography via pixel-pair difference histogram analysis.

STEPS:
1. Load image, use Red channel only (shape: 512, 512)
2. Flatten to 1D array: pixels = channel.flatten()
3. Create non-overlapping pairs: 
   p1_array = pixels[0::2]  (indices 0, 2, 4, ...)
   p2_array = pixels[1::2]  (indices 1, 3, 5, ...)
4. Compute differences: diffs = np.abs(p2_array.astype(int) - p1_array.astype(int))
5. Build PPDH: counts = np.bincount(diffs, minlength=256)
   This gives counts[d] = number of pairs with difference d

6. Detect step discontinuities at PVD quantization boundaries:
   Boundary positions: [7, 15, 31, 63, 127]
   
   For each boundary position k:
   second_derivative = counts[k+1] - 2*counts[k] + counts[k-1]
   
   Also compute local step magnitude:
   left_avg = np.mean(counts[max(0,k-3):k])
   right_avg = np.mean(counts[k+1:min(255,k+4)])
   step_magnitude = abs(right_avg - left_avg)

7. Score PVD likelihood:
   pvd_score = 0
   threshold = np.std(counts) * 2  (adaptive threshold)
   
   For each boundary:
   if abs(second_derivative) > threshold: pvd_score += 1
   
   pvd_detected = pvd_score >= 3  (at least 3 of 5 boundaries show steps)

8. Return:
{
    "pvd_detected": bool,
    "pvd_score": int (0-5, number of boundaries with detected steps),
    "boundary_scores": dict with score at each boundary,
    "confidence": float (pvd_score / 5)
}

FUNCTION 3: validate_and_route(image_path, cnn_prediction, cnn_confidence, threshold=0.70)

This is the main routing function that combines CNN output with statistical tests.

LOGIC:
if cnn_confidence >= threshold:
    if cnn_prediction == 'clean':
        run chi_square test as sanity check
        if lsb_detected: override to 'lsb' with warning logged
        else: confirm 'clean'
    elif cnn_prediction == 'lsb':
        run chi_square_lsb_test
        if lsb_detected: confirm 'lsb', high confidence
        else: flag as uncertain
    elif cnn_prediction == 'pvd':
        run ppdh_pvd_test
        if pvd_detected: confirm 'pvd', high confidence
        else: flag as uncertain
else:
    # Low CNN confidence - run both tests
    lsb_result = chi_square_lsb_test(image_path)
    pvd_result = ppdh_pvd_test(image_path)
    
    if lsb_result['lsb_detected']: final_decision = 'lsb'
    elif pvd_result['pvd_detected']: final_decision = 'pvd'
    else: final_decision = 'clean'

Return:
{
    "final_decision": str ('clean', 'lsb', or 'pvd'),
    "route_to": str ('lsb_extractor', 'pvd_extractor', or 'terminate'),
    "combined_confidence": float,
    "cnn_prediction": str,
    "cnn_confidence": float,
    "statistical_tests": {"chi_square": dict, "ppdh": dict},
    "override_occurred": bool,
    "warning": str or None
}

FUNCTION 4: plot_ppdh(image_path, save_path=None)
Plot the PPDH histogram with vertical lines at boundary positions.
Mark detected steps with red X markers.
Used for visualization and debugging.

When run directly, test all functions on sample images from the dataset 
and print results in a formatted table.

PHASE 6 — MODULE 3 (PAYLOAD EXTRACTOR)
Prompt for coding assistant:
Using the StegoDetect project context, create modules/module3_extractor.py

This module extracts hidden payloads from confirmed stego images.
This is pure algorithmic code - NO machine learning.

FUNCTION 1: extract_lsb(image_path, bits_per_channel=1)

STEPS:
1. Load image:
   img = np.array(Image.open(image_path).convert('RGB'))
   
2. Extract LSBs:
   mask = (2 ** bits_per_channel) - 1
   lsb_array = img & mask
   
3. Flatten in correct order (row by row, R then G then B per pixel):
   bits = lsb_array.flatten()
   
4. If bits_per_channel > 1, need to unpack multi-bit values:
   For bits_per_channel=1: each element is already 0 or 1
   For bits_per_channel=2: each element is 0-3, unpack to 2 bits each
   For bits_per_channel=3: each element is 0-7, unpack to 3 bits each
   
   Use np.unpackbits carefully or manual bit extraction
   
5. Pack bits into bytes:
   byte_array = np.packbits(bits[:len(bits) - (len(bits) % 8)])
   
6. Find null terminator:
   null_positions = np.where(byte_array == 0)[0]
   if len(null_positions) > 0:
       payload_bytes = bytes(byte_array[:null_positions[0]])
   else:
       payload_bytes = bytes(byte_array)
   
7. Return payload_bytes

Handle exceptions:
- If image cannot be opened: raise ExtractionError
- If bits_per_channel not in [1, 2, 3]: raise ValueError

FUNCTION 2: extract_pvd(image_path)

QUANTIZATION TABLE (constant, defined at module level):
PVD_RANGES = [
    (0, 7, 3),
    (8, 15, 3),
    (16, 31, 4),
    (32, 63, 5),
    (64, 127, 6),
    (128, 255, 7)
]

STEPS:
1. Load image:
   img = np.array(Image.open(image_path).convert('RGB'))
   
2. Use Red channel:
   channel = img[:, :, 0].flatten().astype(int)
   
3. Create pairs:
   p1 = channel[0::2]
   p2 = channel[1::2]
   min_len = min(len(p1), len(p2))
   p1, p2 = p1[:min_len], p2[:min_len]
   
4. Compute differences:
   diffs = np.abs(p2 - p1)
   
5. Decode bits from each pair:
   recovered_bits = []
   
   For each pair index i:
       d = diffs[i]
       Find which range d falls in from PVD_RANGES
       n_bits = bits for that range
       range_lower = lower bound of that range
       
       secret_value = d - range_lower
       
       Convert secret_value to n_bits binary digits:
       bits_str = format(secret_value, f'0{n_bits}b')
       recovered_bits.extend([int(b) for b in bits_str])
   
   (Use numpy for efficiency if needed, but ensure correctness first)

6. Pack bits into bytes:
   recovered_bits = recovered_bits[:len(recovered_bits) - (len(recovered_bits) % 8)]
   byte_list = []
   for i in range(0, len(recovered_bits), 8):
       byte_val = int(''.join(map(str, recovered_bits[i:i+8])), 2)
       byte_list.append(byte_val)
   payload_bytes = bytes(byte_list)

7. Find null terminator and return payload_bytes up to it

FUNCTION 3: detect_obfuscation(raw_bytes)
Returns: 'none', 'base64', 'zip', or 'unknown'

Check in this order:
1. ZIP: raw_bytes[:4] == b'PK\x03\x04'
2. Base64: 
   try decode as ASCII
   check all chars are in base64 alphabet: [A-Za-z0-9+/=\n\r]
   try base64.b64decode - if succeeds return 'base64'
3. Return 'none' if neither

FUNCTION 4: deobfuscate(raw_bytes, obfuscation_type=None, max_depth=3)

If obfuscation_type is None, auto-detect.

Recursive deobfuscation up to max_depth layers:
- 'zip': use zipfile module to decompress
- 'base64': use base64.b64decode
- 'none': return as-is

Each step: detect ? reverse ? detect again (for nested obfuscation)

FUNCTION 5: assess_extraction_quality(raw_bytes)

Returns: dict with keys:
- quality: 'good', 'partial', 'failed'  
- printable_ratio: float (percentage of printable characters)
- is_valid_utf8: bool
- likely_language: str or None (quick keyword check)
- byte_length: int

Logic:
- Try decode as UTF-8, check printable_ratio
- quality = 'good' if printable_ratio > 0.85
- quality = 'partial' if 0.50 <= printable_ratio <= 0.85
- quality = 'failed' if printable_ratio < 0.50

For likely_language quick check:
- Contains 'function' or 'eval' or '=>' ? 'javascript'
- Contains 'Invoke-' or '$env:' or 'Get-' ? 'powershell'
- Contains '<script' or '<html' ? 'html'
- Matches ETH regex ? 'ethereum'
- Contains 'http' ? 'url'
- None if uncertain

FUNCTION 6: extract_payload(image_path, technique, confidence=1.0)

MAIN EXTRACTION FUNCTION that calls everything above.

Steps:
1. Try primary extraction based on technique
2. Run assess_extraction_quality on result
3. If quality == 'failed' and confidence < 0.80:
   Try the other technique as fallback
4. Run deobfuscate on best result
5. Run final assess_extraction_quality
6. Return ExtractionResult dataclass

DATACLASS: ExtractionResult
Fields:
- payload_text: str
- payload_bytes_length: int  
- obfuscation_layer: str
- extraction_quality: str
- printable_ratio: float
- technique_used: str
- fallback_used: bool
- likely_language: str
- error: str or None

CUSTOM EXCEPTION: ExtractionError(Exception)

When run directly:
- Test LSB extraction on one image from ./data/combined/test/lsb/
- Test PVD extraction on one image from ./data/combined/test/pvd/
- Print results including payload snippet (first 200 chars)

PHASE 7 — BUILD NLP TRAINING CORPUS
Prompt for coding assistant:
Using the StegoDetect project context, create data_prep/build_nlp_corpus.py

This script builds the training corpus for Module 4 by extracting payloads 
from the LSB training images and labeling them by payload type.

THE CHALLENGE: The LSB dataset's stego images have payloads hidden inside them.
We need to extract those payloads to build our NLP training data.
The payload type for each image is known from the folder structure.

LABEL MAPPING (from folder names):
./data/lsb_dataset/train/stego/  ? folder contains all 5 types mixed
    BUT each image's payload type must be determined from dataset_info.csv if available
    OR from the payload content itself if csv not available

APPROACH:
1. First check if ./data/lsb_dataset/dataset_info.csv exists
   If yes: use it to get payload_type per image filename
   If no: extract payload and auto-label using keyword detection

2. For each image in lsb_dataset/train/stego/:
   a. Extract payload using extract_lsb()
   b. Deobfuscate if needed
   c. Get label from CSV or auto-detect
   d. Store (payload_text, label)

3. Also process stego_b64 and stego_zip:
   a. Extract with extract_lsb()
   b. Deobfuscate (these are definitely obfuscated)
   c. Get label same way
   d. These are important for testing CodeBERT on obfuscated content

AUTO-LABELING FUNCTION: auto_label_payload(payload_text)
Returns label int (0-4) or None if cannot determine

Rules in order (most specific first):
1. ETH address regex: ^0x[0-9a-fA-F]{40}$ ? label 4
2. URL/IP: contains 'http' or matches IP pattern ? label 3
3. PowerShell keywords: 'Invoke-', '$env:', 'Get-Process', 'Set-', 
   'New-Object', '-EncodedCommand' ? label 2
4. HTML+JS: '<script', '<html', '<iframe', 'onclick=' ? label 1
5. JavaScript: 'function', 'eval(', 'document.', '=>', 'var ', 
   'const ', 'let ' ? label 0
6. None if none match

CORPUS BUILDING:
Process train, val, test splits separately.

For each split:
- Extract payloads from lsb/
- Extract payloads from stego_b64/ (mark obfuscation_type='base64')
- Extract payloads from stego_zip/ (mark obfuscation_type='zip')
- Skip images where extraction quality is 'failed'
- Skip payloads with None label (unlabelable)
- Skip payloads shorter than 20 characters

OUTPUT FILES in ./data/nlp_corpus/:
- train.csv with columns: payload_text, label, label_name, obfuscation_type, source_image
- val.csv same
- test.csv same
- corpus_stats.json with:
  {
    "total_samples": int,
    "per_split": {"train": int, "val": int, "test": int},
    "per_class": {"JavaScript": int, "JavaScript_HTML": int, ...},
    "obfuscation_breakdown": {"none": int, "base64": int, "zip": int},
    "extraction_failures": int,
    "auto_labeled": int,
    "csv_labeled": int
  }

SHOW:
- Progress bar per split
- Final class distribution table
- Warning if any class has fewer than 100 samples
- Sample of 3 payloads from each class (first 100 chars)

Run extraction in parallel using ProcessPoolExecutor(max_workers=4) 
for speed since there are 16,000+ images.

PHASE 8 — MODULE 4 (NLP CLASSIFIER)
Prompt for coding assistant:
Using the StegoDetect project context, create modules/module4_nlp.py

This module classifies extracted payload text into 5 categories.
It uses a 3-stage pipeline.

LABEL MAP (constant at module level):
LABEL_NAMES = {
    0: 'JavaScript',
    1: 'JavaScript_HTML', 
    2: 'PowerShell',
    3: 'URL_IP',
    4: 'Ethereum_Address'
}

CLASS: RegexClassifier
Handles Stage 1 - pattern-based classification for ETH and URL.

__init__(self):
    Define regex patterns:
    self.eth_pattern = re.compile(r'^0x[0-9a-fA-F]{40}$')
    self.url_pattern = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|'
        r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?::\d{1,5})?'
    )

classify(self, text):
    text = text.strip()
    if self.eth_pattern.match(text): 
        return 4, 0.99  # label, confidence
    if self.url_pattern.search(text): 
        return 3, 0.97
    return None, 0.0    # None means "cannot classify, try next stage"

CLASS: TFIDFClassifier
Handles Stage 2 - TF-IDF + LinearSVC for JS/HTML/PowerShell.

__init__(self):
    self.pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            strip_accents='unicode'
        )),
        ('svm', LinearSVC(
            C=1.0,
            max_iter=5000,
            class_weight='balanced',
            dual=True
        ))
    ])
    self.is_fitted = False
    self.label_encoder = LabelEncoder()

train(self, texts, labels):
    Fit the pipeline.
    Save trained pipeline to ./models/tfidf_svm.joblib using joblib.dump

classify(self, text):
    if not self.is_fitted: raise RuntimeError("Model not trained")
    prediction = self.pipeline.predict([text])[0]
    
    # Get confidence via decision function distance
    decision = self.pipeline.decision_function([text])[0]
    confidence = float(np.max(softmax(decision)))
    
    return int(prediction), confidence

load(self, path='./models/tfidf_svm.joblib'):
    self.pipeline = joblib.load(path)
    self.is_fitted = True

save(self, path='./models/tfidf_svm.joblib'):
    joblib.dump(self.pipeline, path)

CLASS: CodeBERTClassifier
Handles Stage 3 - fine-tuned CodeBERT for obfuscated payloads.

__init__(self, model_name='microsoft/codebert-base', num_labels=5):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    self.max_length = 512
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(self.device)

classify(self, text):
    self.model.eval()
    inputs = self.tokenizer(
        text,
        truncation=True,
        max_length=self.max_length,
        padding='max_length',
        return_tensors='pt'
    ).to(self.device)
    
    with torch.no_grad():
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, label = torch.max(probs, dim=1)
    
    return int(label.cpu()), float(confidence.cpu())

save(self, path='./models/codebert/'):
    self.model.save_pretrained(path)
    self.tokenizer.save_pretrained(path)

load(self, path='./models/codebert/'):
    self.model = AutoModelForSequenceClassification.from_pretrained(path)
    self.tokenizer = AutoTokenizer.from_pretrained(path)
    self.model.to(self.device)
    self.model.eval()

CLASS: PayloadClassifier
Main NLP classifier that orchestrates all 3 stages.

__init__(self, use_codebert=True):
    self.regex = RegexClassifier()
    self.tfidf_svm = TFIDFClassifier()
    self.codebert = CodeBERTClassifier() if use_codebert else None
    self.use_codebert = use_codebert

classify(self, text, obfuscation_type='none'):
    Stage 1: regex
    label, conf = self.regex.classify(text)
    if label is not None:
        return NLPResult(label, LABEL_NAMES[label], conf, 'regex')
    
    Stage 2: tfidf_svm (always run)
    label_svm, conf_svm = self.tfidf_svm.classify(text)
    
    If obfuscation_type != 'none' and self.use_codebert:
        Stage 3: codebert for obfuscated payloads
        label_bert, conf_bert = self.codebert.classify(text)
        
        if conf_bert > conf_svm:
            return NLPResult(label_bert, LABEL_NAMES[label_bert], conf_bert, 'codebert')
    
    return NLPResult(label_svm, LABEL_NAMES[label_svm], conf_svm, 'tfidf_svm')

DATACLASS: NLPResult
Fields: label(int), label_name(str), confidence(float), classifier_used(str)

load_all(self, models_dir='./models/'):
    Load TF-IDF SVM and CodeBERT from saved files.
    
is_loaded(self):
    Return True if all required models are loaded.

When run directly: test all 5 payload types with sample strings,
print classification results in a table showing predicted label, 
true label, confidence, and classifier used.

PHASE 9 — TRAINING MODULE 4
Prompt for coding assistant:
Using the StegoDetect project context, create training/train_module4.py

This script trains both the TF-IDF+SVM and CodeBERT classifiers for Module 4.

PART A: Train TF-IDF + SVM

1. Load corpus from ./data/nlp_corpus/train.csv
2. Load validation from ./data/nlp_corpus/val.csv

3. Text preprocessing function clean_payload_text(text):
   - Remove null bytes: text.replace('\x00', '')
   - Strip leading/trailing whitespace
   - Truncate to 10,000 characters max (very long payloads cause memory issues)
   - Return cleaned text

4. Train the TFIDFClassifier:
   - Apply clean_payload_text to all texts
   - Fit on training data
   - Evaluate on validation data:
     - Accuracy
     - Weighted F1
     - Per-class F1
     - Confusion matrix
   - Save model to ./models/tfidf_svm.joblib
   - Save confusion matrix plot to ./results/tfidf_confusion_matrix.png

5. Print formatted classification report

PART B: Fine-tune CodeBERT

TRAINING CONFIG:
- Epochs: 5
- Batch size: 16
- Learning rate: 2e-5
- Warmup steps: 10% of total training steps
- Weight decay: 0.01
- Optimizer: AdamW from transformers
- Scheduler: get_linear_schedule_with_warmup
- Max sequence length: 512
- Save best model based on validation F1

TRAINING LOOP:
For each epoch:
    Train:
    - Load batch of (input_ids, attention_mask, labels)
    - Forward pass through CodeBERT
    - Compute cross-entropy loss
    - Backward pass
    - Clip gradients (max_norm=1.0)
    - Update with optimizer and scheduler
    - Log batch loss every 50 steps
    
    Validate:
    - Compute loss, accuracy, weighted F1
    - If best F1 so far: save model to ./models/codebert/

IMPORTANT: Only train CodeBERT on obfuscated samples:
   Filter corpus to only rows where obfuscation_type in ['base64', 'zip']
   This focuses CodeBERT's specialty on the hard cases that need it.
   If fewer than 500 obfuscated samples: use all samples.

EVALUATION:
After training both models:
1. Run evaluation on test set
2. Test each model separately:
   - TF-IDF+SVM on non-obfuscated test samples
   - CodeBERT on obfuscated test samples
3. Test combined PayloadClassifier on all test samples
4. Save complete metrics to ./results/module4_metrics.json

COMMAND LINE:
python train_module4.py --skip_codebert (to train only TF-IDF+SVM)
                        --epochs 5 
                        --batch_size 16
                        --device cuda

PHASE 10 — END-TO-END PIPELINE
Prompt for coding assistant:
Using the StegoDetect project context, create pipeline.py

This is the main inference pipeline that runs a complete image through 
all 4 modules and produces the final report.

DATACLASS: StegoReport
Fields:
- image_path: str
- image_filename: str
- technique_detected: str  ('clean', 'lsb', 'pvd', 'uncertain')
- detection_confidence: float
- statistical_confirmation: bool
- override_occurred: bool
- extraction_attempted: bool
- extraction_success: bool
- extraction_quality: str
- obfuscation_layer: str
- payload_snippet: str  (first 300 chars of payload)
- payload_full_length: int
- payload_category: str or None
- category_confidence: float or None
- classifier_used: str or None
- risk_level: str  ('none', 'low', 'medium', 'high', 'critical')
- processing_time_ms: float
- warnings: list of str
- error: str or None

FUNCTION: compute_risk_level(technique, payload_category, detection_confidence)

Risk matrix:
technique='clean' ? 'none'
technique='uncertain' ? 'low'

technique in ['lsb', 'pvd'] with payload_category:
    'PowerShell' ? 'critical'
    'JavaScript' ? 'high'
    'JavaScript_HTML' ? 'high'  
    'URL_IP' ? 'medium' if technique=='lsb' else 'high'
    'Ethereum_Address' ? 'medium'
    None (extraction failed) ? 'medium'

If detection_confidence < 0.70: downgrade by one level

CLASS: StegoDetectPipeline

__init__(self, 
         module1_checkpoint='./checkpoints/module1/best_model.pth',
         models_dir='./models/',
         device='auto',
         use_codebert=True,
         confidence_threshold=0.70):
    
    Set device (auto = cuda if available)
    Load Module 1: StegoClassifier ? eval mode
    Load Module 4: PayloadClassifier (tfidf_svm + optionally codebert)
    Module 2 and 3 are stateless functions, no loading needed.

analyze(self, image_path):
    
    Start timer.
    
    STEP 1 — Module 1 (CNN Detection):
    Load and preprocess image (resize to 224x224, normalize)
    Run through StegoClassifier
    Get: predicted_class (0,1,2), confidence, all_probabilities
    Map class to string: {0:'clean', 1:'lsb', 2:'pvd'}
    
    STEP 2 — Module 2 (Statistical Validation):
    Call validate_and_route(image_path, cnn_prediction, cnn_confidence, threshold)
    Get: final_decision, route_to, statistical confirmation
    
    If route_to == 'terminate':
        Stop here. Return report with technique='clean', no extraction.
    
    STEP 3 — Module 3 (Payload Extraction):
    Call extract_payload(image_path, technique=final_decision)
    Get: ExtractionResult with payload_text, quality, obfuscation_type
    
    If extraction_quality == 'failed':
        Return report with extraction_success=False
    
    STEP 4 — Module 4 (NLP Classification):
    Call payload_classifier.classify(payload_text, obfuscation_type)
    Get: NLPResult with label, label_name, confidence, classifier_used
    
    STEP 5 — Compute risk level and assemble report.
    
    Stop timer.
    Return StegoReport dataclass.

analyze_batch(self, image_paths, max_workers=4):
    Process multiple images in parallel using ThreadPoolExecutor.
    Show progress bar.
    Return list of StegoReport objects.
    Also return summary dict:
    {
        "total": int,
        "clean": int,
        "lsb_detected": int,
        "pvd_detected": int,
        "critical": int,
        "high": int,
        "medium": int,
        "failed": int
    }

FUNCTION: format_report(report: StegoReport) ? str
Return a nicely formatted string showing all report fields.
Use colors if terminal supports it (use colorama library if available).

FUNCTION: report_to_dict(report: StegoReport) ? dict
Convert to JSON-serializable dictionary.

FUNCTION: save_report(report, output_dir='./results/reports/')
Save as JSON file named {image_filename}_{timestamp}.json

When run directly:
python pipeline.py --image path/to/image.png
python pipeline.py --folder path/to/folder/ --output ./results/
python pipeline.py --test (run on 5 images from test set and show reports)

PHASE 11 — EVALUATION SUITE
Prompt for coding assistant:
Using the StegoDetect project context, create evaluate.py

This script runs comprehensive evaluation of the full pipeline.

FUNCTION 1: evaluate_module1(model, test_loader, device)

Runs evaluation of just Module 1 CNN on the combined test set.

Collect all predictions and true labels.
Compute and print:
- Overall accuracy
- Per-class accuracy (clean, lsb, pvd)
- Per-class precision, recall, F1
- Macro F1, Weighted F1
- Confusion matrix (print as table AND save as PNG heatmap)
- FALSE POSITIVE RATE for clean images specifically:
  FPR = FP_clean / (FP_clean + TN_clean)
  Print warning if FPR > 0.02 (our 2% target)
- ROC AUC per class (one-vs-rest)

Save all metrics to ./results/module1_eval.json

FUNCTION 2: evaluate_extraction(pipeline, test_images_lsb, test_images_pvd)

Test Module 3 extraction accuracy.
For each image in test set:
- Extract payload
- Measure extraction_quality
- For stego images: verify extracted text is valid UTF-8

Compute:
- LSB extraction success rate
- PVD extraction success rate  
- Base64 deobfuscation success rate
- ZIP deobfuscation success rate

Save to ./results/module3_eval.json

FUNCTION 3: evaluate_module4(classifier, test_corpus_path)

Load test corpus from ./data/nlp_corpus/test.csv

Evaluate on three subsets:
1. All samples (full test set)
2. Non-obfuscated only (obfuscation_type='none')
3. Obfuscated only (obfuscation_type in ['base64', 'zip'])

For each subset compute:
- Accuracy, weighted F1, macro F1
- Per-class precision, recall, F1
- Confusion matrix

Compare TF-IDF SVM vs CodeBERT performance on obfuscated subset.

Save to ./results/module4_eval.json

FUNCTION 4: evaluate_end_to_end(pipeline, test_dir)

The most important evaluation - full pipeline from image to report.

Test images from ./data/combined/test/:
- 50 random images from clean/
- 50 random images from lsb/
- 50 random images from pvd/

For each image:
- Know the true label (from folder)
- Run full pipeline.analyze()
- Compare final_decision to true label

Compute:
- End-to-end accuracy (correct technique + correct payload type)
- Detection accuracy (just technique correct)
- Average processing time per image
- Success rate by risk level

Also test edge cases:
- Very small images (resized to 512x512)
- Images with very short payloads
- Images where extraction quality is 'partial'

Save comprehensive report to ./results/end_to_end_eval.json

FUNCTION 5: generate_evaluation_report()

Creates a human-readable summary report in markdown format.
Saved to ./results/EVALUATION_REPORT.md

Includes:
- Executive summary with key metrics
- Per-module results
- Identified weaknesses
- Comparison to baseline (chi-square only)

COMMAND LINE:
python evaluate.py --module all          (run all evaluations)
python evaluate.py --module 1            (module 1 only)
python evaluate.py --module 4            (module 4 only)
python evaluate.py --module end_to_end   (full pipeline only)
python evaluate.py --quick               (sample of 20 images per class)

PHASE 12 — GRADIO WEB INTERFACE
Prompt for coding assistant:
Using the StegoDetect project context, create app.py

Build a Gradio web interface for StegoDetect.

INTERFACE LAYOUT:

Title: "StegoDetect — Steganography Detection & Payload Analysis"
Description: One paragraph explaining what the tool does.

Tab 1: "Analyze Image"
Input:
- Image upload component (accepts PNG, JPG - auto-converts to PNG internally)
- Checkbox: "Show technical details" (default: unchecked)

Output section (appears after analysis):
1. DETECTION RESULT - large colored badge:
   Green = Clean, Orange = Uncertain, Red = LSB/PVD Detected
   Text: "LSB Steganography Detected — 94% confidence"

2. RISK LEVEL badge (colored pill):
   Green=None, Yellow=Low, Orange=Medium, Red=High, Dark Red=Critical

3. If stego detected - show extraction results:
   - Technique: "LSB (Least Significant Bit)"
   - Obfuscation: "ZIP Compressed"
   - Payload Category: "PowerShell Script"
   - Category Confidence: "97%"
   - Payload Preview: Scrollable text box (first 500 chars, monospace font)

4. If "Show technical details" checked - show:
   - CNN prediction: {clean: 0.04, lsb: 0.93, pvd: 0.03}
   - Statistical validation: Chi-square p-value, PPDH score
   - Extraction quality: "good (98.7% printable)"
   - Processing time: "127ms"
   - Classifier used: "TF-IDF + SVM"

5. Download button: Download JSON report

Tab 2: "Batch Analysis"  
Input:
- File upload (accept multiple images or a ZIP file)
- Max 20 images at once

Output:
- Progress bar during processing
- Summary table showing: Filename | Technique | Risk Level | Payload Type
- Download CSV report button
- Summary statistics: X clean, Y LSB detected, Z PVD detected

Tab 3: "About"
Static content explaining:
- What is steganography
- LSB vs PVD explanation
- Project details
- Dataset information
- Model information

FUNCTIONS:
analyze_single_image(image, show_technical):
    If image is None: return "Please upload an image"
    Save image temporarily as PNG
    Run pipeline.analyze()
    Format results for display
    Return all output components

analyze_batch(files):
    Process each file
    Return summary table and CSV

ERROR HANDLING:
- If models not loaded: show "Models not loaded. Run training first."
- If image too small: "Image must be at least 100x100 pixels"
- If not PNG/JPG: "Only PNG and JPG images are supported"
- If pipeline fails: show error message with details

LAUNCH:
if __name__ == "__main__":
    Load pipeline (with error handling if models don't exist)
    Launch with:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

PHASE 13 — RUN ORDER AND INTEGRATION TEST
Prompt for coding assistant:
Using the StegoDetect project context, create run_pipeline.sh and 
a complete integration test.

CREATE: run_pipeline.sh
A bash script that runs the entire project setup in correct order:

#!/bin/bash
set -e  # Exit on any error

echo "=== StegoDetect Setup Pipeline ==="

echo "Step 1: Installing dependencies"
pip install -r requirements.txt

echo "Step 2: Initializing project structure"
python config.py

echo "Step 3: Verifying raw datasets"
python data_prep/verify_dataset.py

echo "Step 4: Merging datasets"
python data_prep/merge_datasets.py

echo "Step 5: Verifying merged dataset"
python data_prep/verify_dataset.py --dir ./data/combined/

echo "Step 6: Building NLP corpus"
python data_prep/build_nlp_corpus.py

echo "Step 7: Training Module 1 (CNN)"
python training/train_module1.py --epochs 50 --device cuda

echo "Step 8: Training Module 4 (NLP)"
python training/train_module4.py --device cuda

echo "Step 9: Running full evaluation"
python evaluate.py --module all

echo "Step 10: Launching web interface"
python app.py

echo "=== Setup Complete ==="

CREATE: tests/integration_test.py

Test that the entire pipeline works end-to-end with dummy data.

Tests to include:

test_dataset_structure():
    Check ./data/combined/ has correct structure
    Check at least 1 image exists in each class/split folder

test_module1_loads():
    Create StegoClassifier instance
    Run forward pass with random 224x224x3 tensor
    Assert output shape is (1, 3)
    Assert probabilities sum to 1.0

test_chi_square():
    Load a known clean image from test set
    Run chi_square_lsb_test
    Assert lsb_detected is False (clean image should not trigger)
    
    Load a known LSB stego image
    Run chi_square_lsb_test
    Assert lsb_detected is True

test_lsb_extraction():
    Load a known LSB stego image from test set
    Run extract_lsb
    Assert result is not empty
    Assert extraction_quality is 'good'

test_pvd_extraction():
    Load a known PVD stego image from test set
    Run extract_pvd
    Assert result is not empty
    Assert extraction_quality != 'failed'

test_regex_classifier():
    eth_addr = "0x71C7656EC7ab88b098defB751B7401B5f6d8976F"
    label, conf = RegexClassifier().classify(eth_addr)
    assert label == 4
    assert conf > 0.95
    
    url = "http://malicious-site.com/payload"
    label, conf = RegexClassifier().classify(url)
    assert label == 3

test_full_pipeline():
    If models are loaded:
        Pick one image from each class in test set
        Run pipeline.analyze() on each
        Assert report is StegoReport instance
        Assert processing_time_ms > 0
        Assert risk_level is not None

Run all tests with:
python -m pytest tests/integration_test.py -v

COMMON ERRORS AND FIXES
Paste this if your coding assistant encounters issues:
COMMON ISSUES IN STEGODETECT PROJECT:

ISSUE 1: CUDA out of memory during Module 1 training
FIX: Reduce batch_size from 32 to 16. 
Add torch.cuda.empty_cache() after each epoch.
Use gradient_checkpointing=True in model config.

ISSUE 2: NaN loss during training
FIX: Reduce learning rate by 10x.
Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
Check for NaN in input images (normalize properly).

ISSUE 3: PVD extraction produces garbage
FIX: Verify non-overlapping pairs: p1=pixels[0::2], p2=pixels[1::2]
Do NOT use p1=pixels[:-1:2] (this is wrong)
Verify quantization table boundaries match embedding tool exactly.

ISSUE 4: Chi-square test flags clean images
FIX: Adjust p-value threshold from 0.05 to 0.01 for stricter detection.
Run on all 3 channels separately and require at least 2/3 to trigger.

ISSUE 5: TF-IDF gets less than 80% accuracy
FIX: Try ngram_range=(1,4) instead of (2,4).
Increase max_features to 100000.
Try C=0.1 or C=10 for LinearSVC.
Check class balance - use class_weight='balanced'.

ISSUE 6: Module 1 overfits (train accuracy >> val accuracy)
FIX: Increase dropout from 0.3 to 0.5.
Freeze more backbone layers during Phase 1.
Add weight_decay=1e-4.
Reduce epochs.

ISSUE 7: LSB extraction successful but NLP classification wrong
FIX: Check deobfuscation worked correctly.
Print first 100 chars of payload_text to verify it looks like real code.
Check label mapping in corpus matches training.

ISSUE 8: Image loading fails
FIX: Always use Image.open(path).convert('RGB') to handle RGBA or grayscale.
Never assume image mode.

ISSUE 9: Filename conflicts in merged dataset
FIX: Use full path hash as part of filename: 
new_name = f"{source_prefix}_{hashlib.md5(original_path.encode()).hexdigest()[:8]}_{original_name}"

ISSUE 10: CodeBERT runs out of GPU memory
FIX: Reduce batch_size to 8.
Use fp16 training: scaler = torch.cuda.amp.GradScaler()
Use gradient_accumulation_steps=4 to simulate batch_size=32.

QUICK REFERENCE — EXECUTION ORDER
ONE-TIME SETUP (run in this exact order):
1.  pip install -r requirements.txt
2.  python config.py
3.  python data_prep/verify_dataset.py
4.  python data_prep/merge_datasets.py
5.  python data_prep/build_nlp_corpus.py
6.  python training/train_module1.py
7.  python training/train_module4.py
8.  python evaluate.py --module all

DAILY USE (after setup):
    python pipeline.py --image your_image.png
    python app.py  (web interface)

TESTING:
    python -m pytest tests/integration_test.py -v
    python pipeline.py --test

FILE OUTPUTS TO EXPECT:
./checkpoints/module1/best_model.pth     ? CNN weights
./models/tfidf_svm.joblib                ? SVM model
./models/codebert/                       ? CodeBERT weights
./data/combined/                         ? Merged dataset
./data/nlp_corpus/train.csv             ? NLP training data
./results/EVALUATION_REPORT.md          ? Final metrics

