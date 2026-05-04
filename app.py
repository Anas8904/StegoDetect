import os
import sys
import json
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import gradio as gr

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from modules.module1_cnn import StegoClassifier, load_pretrained_stego
from modules.dataset import get_transforms


# ──────────────────────────────────────────────────────────────────────
# Global state: load models on startup
# ──────────────────────────────────────────────────────────────────────
PIPELINE_LOADED = False
MODEL = None
TRANSFORM = None
DECISION_THRESHOLDS = None   # calibrated thresholds from training


def _add_noise_residual(image_tensor: torch.Tensor, pil_image: Image.Image) -> torch.Tensor:
    """
    Add Laplacian noise-residual as 4th channel, matching dataset.py logic.

    This is required when the model was trained with use_noise_residual=True.
    """
    img_np = np.array(pil_image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)

    lap_min, lap_max = laplacian.min(), laplacian.max()
    if lap_max - lap_min > 0:
        laplacian_norm = (laplacian - lap_min) / (lap_max - lap_min)
    else:
        laplacian_norm = np.zeros_like(laplacian)

    size = config.CNN_CONFIG["input_size"]
    laplacian_resized = cv2.resize(laplacian_norm, (size, size))
    noise_channel = torch.tensor(laplacian_resized, dtype=torch.float32).unsqueeze(0)
    return torch.cat([image_tensor, noise_channel], dim=0)


def load_models():
    """Load the CNN model checkpoint and decision thresholds."""
    global PIPELINE_LOADED, MODEL, TRANSFORM, DECISION_THRESHOLDS

    checkpoint_path = config.MODULE1_CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"[WARN] Model not found at {checkpoint_path}")
        print("   Run training first: python training/train_module1.py")
        return False

    try:
        MODEL = load_pretrained_stego(str(checkpoint_path), device=config.DEVICE)
        TRANSFORM = get_transforms("test")

        # Load calibrated thresholds if available
        thresh_path = config.MODULE1_CHECKPOINT_DIR / "decision_thresholds.json"
        if thresh_path.exists():
            with open(thresh_path) as f:
                raw = json.load(f)
            # Keys are stored as strings in JSON — convert back to int
            DECISION_THRESHOLDS = {int(k): v for k, v in raw.items()}
            print(f"[OK] Decision thresholds loaded: {DECISION_THRESHOLDS}")
        else:
            DECISION_THRESHOLDS = None
            print("[INFO] No decision thresholds found — using argmax.")

        PIPELINE_LOADED = True
        print("[OK] Models loaded successfully!")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load models: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────────────────────────────

def analyze_single_image(image, show_technical):
    """
    Analyze a single image for steganography.

    Returns formatted results for display in the Gradio UI.
    """
    if image is None:
        return (
            "⚠️ Please upload an image",   # detection result
            "",                              # risk level
            "",                              # details
            "",                              # technical details
            None,                            # JSON report
        )

    if not PIPELINE_LOADED:
        return (
            "⚠️ Models not loaded. Run training first:\n  python training/train_module1.py",
            "", "", "", None,
        )

    start_time = time.time()

    try:
        # ── Preprocess ────────────────────────────────────────────────
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # Check minimum size
        w, h = pil_image.size
        if w < 100 or h < 100:
            return ("⚠️ Image must be at least 100×100 pixels", "", "", "", None)

        # Transform and predict
        input_tensor = TRANSFORM(pil_image)
        # Add noise-residual 4th channel if the model was trained with it
        if MODEL.use_noise_residual:
            input_tensor = _add_noise_residual(input_tensor, pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)
        pred_class, confidence, probs = MODEL.predict_with_confidence(
            input_tensor, thresholds=DECISION_THRESHOLDS,
        )

        pred_idx = pred_class.item()
        conf = confidence.item()
        prob_dict = {
            name: float(probs[0][i].item())
            for i, name in enumerate(config.CLASS_NAMES)
        }

        processing_time = (time.time() - start_time) * 1000  # ms

        # ── Format detection result ───────────────────────────────────
        class_name = config.CLASS_NAMES[pred_idx]
        if class_name == "clean":
            detection_text = f"✅ CLEAN — No steganography detected ({conf*100:.1f}% confidence)"
            risk_level = "🟢 Risk Level: None"
            details = "No hidden data was found in this image."
        elif class_name == "lsb":
            detection_text = f"🔴 LSB Steganography Detected — {conf*100:.1f}% confidence"
            risk_level = "🟠 Risk Level: High"
            details = (
                f"**Technique:** LSB (Least Significant Bit)\n\n"
                f"The CNN model has detected LSB steganography patterns in this image. "
                f"Hidden data may be embedded in the least significant bits of pixel values."
            )
        else:  # pvd
            detection_text = f"🔴 PVD Steganography Detected — {conf*100:.1f}% confidence"
            risk_level = "🟠 Risk Level: High"
            details = (
                f"**Technique:** PVD (Pixel Value Differencing)\n\n"
                f"The CNN model has detected PVD steganography patterns in this image. "
                f"Hidden data may be embedded using pixel pair differences."
            )

        # ── Technical details ─────────────────────────────────────────
        if show_technical:
            tech_text = (
                f"**CNN Prediction Probabilities:**\n"
                f"```\n"
                f"  clean:  {prob_dict['clean']:.4f}\n"
                f"  lsb:    {prob_dict['lsb']:.4f}\n"
                f"  pvd:    {prob_dict['pvd']:.4f}\n"
                f"```\n\n"
                f"**Processing time:** {processing_time:.0f} ms\n\n"
                f"**Device:** {config.DEVICE}\n\n"
                f"**Image size:** {w}×{h} pixels"
            )
        else:
            tech_text = ""

        # ── JSON report ───────────────────────────────────────────────
        report = {
            "detection": class_name,
            "confidence": round(conf, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "processing_time_ms": round(processing_time, 1),
            "image_size": [w, h],
        }

        # Save report to temp file for download
        report_path = Path(tempfile.gettempdir()) / "stegodetect_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        return (detection_text, risk_level, details, tech_text, str(report_path))

    except Exception as e:
        return (f"❌ Error: {str(e)}", "", "", "", None)


def analyze_batch(files):
    """Analyze multiple images and return a summary table."""
    if not files:
        return "⚠️ Please upload images", None

    if not PIPELINE_LOADED:
        return "⚠️ Models not loaded. Run training first.", None

    results = []
    for file_obj in files:
        try:
            img = Image.open(file_obj.name).convert("RGB")
            input_tensor = TRANSFORM(img)
            if MODEL.use_noise_residual:
                input_tensor = _add_noise_residual(input_tensor, img)
            input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)
            pred_class, confidence, probs = MODEL.predict_with_confidence(
                input_tensor, thresholds=DECISION_THRESHOLDS,
            )

            pred_idx = pred_class.item()
            conf = confidence.item()
            class_name = config.CLASS_NAMES[pred_idx]

            risk = "None" if class_name == "clean" else "High"

            results.append({
                "Filename": Path(file_obj.name).name,
                "Technique": class_name.upper(),
                "Confidence": f"{conf*100:.1f}%",
                "Risk Level": risk,
            })
        except Exception as e:
            results.append({
                "Filename": Path(file_obj.name).name,
                "Technique": "ERROR",
                "Confidence": "-",
                "Risk Level": str(e),
            })

    # Summary
    clean_count = sum(1 for r in results if r["Technique"] == "CLEAN")
    lsb_count = sum(1 for r in results if r["Technique"] == "LSB")
    pvd_count = sum(1 for r in results if r["Technique"] == "PVD")
    summary = (
        f"**Results:** {len(results)} images analyzed\n\n"
        f"🟢 Clean: {clean_count}  |  🔴 LSB: {lsb_count}  |  🔴 PVD: {pvd_count}"
    )

    # Build table
    import pandas as pd
    df = pd.DataFrame(results)

    return summary, df


# ──────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
}
.main-title {
    text-align: center;
    color: #1a1a2e;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-top: 0;
}
"""

ABOUT_TEXT = """
## What is Steganography?

**Steganography** is the practice of hiding secret data within ordinary, non-secret data
(such as images) to avoid detection. Unlike encryption, which makes data unreadable,
steganography hides the very existence of the secret message.

## Detection Techniques

### LSB (Least Significant Bit)
The most common image steganography technique. Data is hidden by replacing the least
significant bits of pixel values. The visual change is imperceptible to the human eye,
but can be detected statistically.

### PVD (Pixel Value Differencing)
A more advanced technique that embeds data in the differences between adjacent pixel pairs.
Larger differences can hide more bits, making it harder to detect in textured regions.

## About StegoDetect

StegoDetect is a 4-module AI-powered pipeline:

1. **Module 1 — CNN Detection:** EfficientNet-B0 classifies images as clean, LSB, or PVD
2. **Module 2 — Statistical Validation:** Chi-square and PPDH tests confirm findings
3. **Module 3 — Payload Extraction:** Reverses the embedding algorithm to extract hidden data
4. **Module 4 — NLP Classification:** Classifies extracted payloads (PowerShell, JavaScript, URLs, etc.)

### Tech Stack
- **Model:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned on steganography dataset)
- **Framework:** PyTorch 2.1+
- **Dataset:** 60,000 images (512×512 PNG) — clean, LSB, and PVD steganography
- **NLP:** TF-IDF + SVM, with optional CodeBERT for obfuscated payloads

### Dataset Information
| Split | Clean | LSB | PVD | Total |
|-------|-------|-----|-----|-------|
| Train | 8,000 | 12,000 | 3,995 | 23,995 |
| Test | 4,000 | 18,000 | 2,000 | 24,000 |
| Val | 4,000 | 6,000 | 2,000 | 12,000 |
"""


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="StegoDetect — Steganography Detection & Payload Analysis",
    ) as demo:
        gr.Markdown("# 🔍 StegoDetect", elem_classes=["main-title"])
        gr.Markdown(
            "AI-Powered Steganography Detection & Payload Analysis",
            elem_classes=["subtitle"],
        )

        # ── Tab 1: Analyze Image ─────────────────────────────────────
        with gr.Tab("🖼️ Analyze Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=350,
                    )
                    show_tech = gr.Checkbox(
                        label="Show technical details",
                        value=False,
                    )
                    analyze_btn = gr.Button(
                        "🔍 Analyze",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=1):
                    detection_output = gr.Markdown(label="Detection Result")
                    risk_output = gr.Markdown(label="Risk Level")
                    details_output = gr.Markdown(label="Details")
                    tech_output = gr.Markdown(label="Technical Details")
                    report_download = gr.File(label="📥 Download JSON Report")

            analyze_btn.click(
                fn=analyze_single_image,
                inputs=[image_input, show_tech],
                outputs=[detection_output, risk_output, details_output, tech_output, report_download],
            )

        # ── Tab 2: Batch Analysis ────────────────────────────────────
        with gr.Tab("📊 Batch Analysis"):
            gr.Markdown("Upload multiple images (max 20) for bulk analysis.")
            file_input = gr.Files(
                label="Upload Images",
                file_types=["image"],
                file_count="multiple",
            )
            batch_btn = gr.Button("🔍 Analyze Batch", variant="primary")

            batch_summary = gr.Markdown(label="Summary")
            batch_table = gr.Dataframe(
                label="Results",
                headers=["Filename", "Technique", "Confidence", "Risk Level"],
            )

            batch_btn.click(
                fn=analyze_batch,
                inputs=[file_input],
                outputs=[batch_summary, batch_table],
            )

        # ── Tab 3: About ─────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT_TEXT)

    return demo


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Try loading models (will warn if not trained yet)
    loaded = load_models()
    if not loaded:
        print("\n⚠  Starting UI without models — analysis will show error messages.")
        print("   Train the model first: python training/train_module1.py\n")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css=CUSTOM_CSS,
    )
