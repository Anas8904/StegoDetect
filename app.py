"""
StegoDetect - Gradio Web Interface
AI-Powered Steganography Detection & Payload Analysis

Tab 1: Analyze Image  - Full 4-module pipeline on single image
Tab 2: Batch Analysis  - Bulk analysis of multiple images
Tab 3: About           - Project documentation & architecture
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image
import gradio as gr

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from modules.module1_cnn import StegoClassifier, load_pretrained_stego
from modules.dataset import get_transforms
from modules.module2_validator import chi_square_lsb_test, ppdh_pvd_test, validate_and_route
from modules.module3_extractor import extract_payload, ExtractionResult
from modules.module4_nlp import PayloadClassifier, NLPResult, LABEL_NAMES


# ──────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────
PIPELINE_LOADED = False
MODEL = None
TRANSFORM = None
NLP_CLASSIFIER = None


def load_models():
    """Load CNN model and NLP classifier."""
    global PIPELINE_LOADED, MODEL, TRANSFORM, NLP_CLASSIFIER

    checkpoint_path = config.MODULE1_CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"[WARN] CNN model not found at {checkpoint_path}")
        print("   Train first: python training/train_module1.py")
        return False

    try:
        MODEL = load_pretrained_stego(str(checkpoint_path), device=config.DEVICE)
        TRANSFORM = get_transforms("test")
        NLP_CLASSIFIER = PayloadClassifier(use_codebert=False)
        NLP_CLASSIFIER.load_all()
        PIPELINE_LOADED = True
        print("[OK] All models loaded successfully!")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load models: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# SRM Kernels for noise residual
# ──────────────────────────────────────────────────────────────────────
_SRM_K1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float64)
_SRM_K2 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], dtype=np.float64)
_SRM_K3 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float64)


def preprocess_image(pil_image):
    image_tensor = TRANSFORM(pil_image)
    if getattr(MODEL, "use_noise_residual", False):
        size = config.CNN_CONFIG["input_size"]
        img_resized = pil_image.resize((size, size), Image.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float64)
        residuals = []
        for kernel in [_SRM_K1, _SRM_K2, _SRM_K3]:
            ch_res = [cv2.filter2D(img_np[:,:,c], cv2.CV_64F, kernel) for c in range(3)]
            residuals.append(np.mean(ch_res, axis=0))
        noise_map = np.clip(np.mean(residuals, axis=0) / 4.0, -3.0, 3.0)
        noise_channel = torch.tensor(noise_map, dtype=torch.float32).unsqueeze(0)
        image_tensor = torch.cat([image_tensor, noise_channel], dim=0)
    return image_tensor.unsqueeze(0).to(config.DEVICE)


# ──────────────────────────────────────────────────────────────────────
# Risk mapping helper
# ──────────────────────────────────────────────────────────────────────
def compute_risk(technique, payload_category, confidence):
    if technique == "clean":
        return "None", "#10b981"
    risk_map = {"PowerShell": "Critical", "JavaScript": "High",
                "JavaScript_HTML": "High", "URL_IP": "Medium", "Ethereum_Address": "Medium"}
    risk = risk_map.get(payload_category, "Medium") if payload_category else "Medium"
    if confidence < 0.70:
        downgrade = {"Critical": "High", "High": "Medium", "Medium": "Low"}
        risk = downgrade.get(risk, risk)
    colors = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#3b82f6", "None": "#10b981"}
    return risk, colors.get(risk, "#6b7280")


# ──────────────────────────────────────────────────────────────────────
# Single image analysis
# ──────────────────────────────────────────────────────────────────────
def analyze_single_image(image, show_technical):
    if image is None:
        return "Upload an image to begin analysis.", "", "", "", "", None

    if not PIPELINE_LOADED:
        return ("[WARN] Models not loaded. Train first:\n"
                "  python training/train_module1.py --epochs 50 --batch_size 16 --device cuda",
                "", "", "", "", None)

    start_time = time.time()
    try:
        pil_image = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")
        w, h = pil_image.size
        if w < 100 or h < 100:
            return "[WARN] Image must be at least 100x100 pixels.", "", "", "", "", None

        temp_path = Path(tempfile.gettempdir()) / "stegodetect_temp_input.png"
        pil_image.save(str(temp_path), format="PNG")

        # MODULE 1: CNN
        input_tensor = preprocess_image(pil_image)
        pred_class, confidence, probs = MODEL.predict_with_confidence(input_tensor)
        pred_idx = pred_class.item()
        conf = confidence.item()
        class_name = config.CLASS_NAMES[pred_idx]
        prob_dict = {name: float(probs[0][i].item()) for i, name in enumerate(config.CLASS_NAMES)}

        # MODULE 2: Statistical Validation
        validation = validate_and_route(str(temp_path), cnn_prediction=class_name, cnn_confidence=conf)
        final_decision = validation["final_decision"]
        combined_conf = validation["combined_confidence"]

        # MODULE 3: Extraction
        extraction_result = None
        if final_decision in ("lsb", "pvd"):
            extraction_result = extract_payload(str(temp_path), technique=final_decision, confidence=combined_conf)

        # MODULE 4: NLP Classification
        nlp_result = None
        if extraction_result and extraction_result.extraction_quality != "failed" and extraction_result.payload_text:
            nlp_result = NLP_CLASSIFIER.classify(
                extraction_result.payload_text,
                obfuscation_type=extraction_result.obfuscation_layer,
            )

        processing_time = (time.time() - start_time) * 1000

        # Risk level
        payload_cat = nlp_result.label_name if nlp_result and nlp_result.label != -1 else None
        risk_label, risk_color = compute_risk(final_decision, payload_cat, combined_conf)

        # Format detection result
        icons = {"clean": "&#9989;", "lsb": "&#9888;&#65039;", "pvd": "&#9888;&#65039;"}
        icon = icons.get(final_decision, "")
        if final_decision == "clean":
            detection_text = f"### {icon} CLEAN\nNo steganography detected ({combined_conf*100:.1f}% confidence)"
        else:
            tech_name = "LSB (Least Significant Bit)" if final_decision == "lsb" else "PVD (Pixel Value Differencing)"
            detection_text = f"### {icon} {final_decision.upper()} Steganography Detected\n**Technique:** {tech_name}\n**Confidence:** {combined_conf*100:.1f}%"

        risk_text = f"### Risk Level: <span style='color:{risk_color};font-weight:bold'>{risk_label}</span>"

        # Validation & extraction details
        detail_parts = []
        chi_result = validation["statistical_tests"]["chi_square"]
        pvd_stat = validation["statistical_tests"]["ppdh"]

        if validation["override_occurred"]:
            detail_parts.append(f"> **Override:** {validation['warning']}\n")
        elif validation["warning"]:
            detail_parts.append(f"> **Note:** {validation['warning']}\n")

        detail_parts.append("#### Module 2 - Statistical Validation")
        if chi_result:
            status = "Detected" if chi_result['lsb_detected'] else "Not detected"
            detail_parts.append(f"- Chi-Square LSB: **{status}** (p={chi_result['p_value']:.4f})")
        if pvd_stat:
            status = "Detected" if pvd_stat['pvd_detected'] else "Not detected"
            detail_parts.append(f"- PPDH PVD: **{status}** (score={pvd_stat['pvd_score']}/5)")

        if extraction_result and extraction_result.extraction_quality != "failed":
            detail_parts.append("\n#### Module 3 - Payload Extraction")
            detail_parts.append(f"- Quality: **{extraction_result.extraction_quality}**")
            detail_parts.append(f"- Printable ratio: {extraction_result.printable_ratio:.1%}")
            detail_parts.append(f"- Size: {extraction_result.payload_bytes_length} bytes")
            if extraction_result.obfuscation_layer != "none":
                detail_parts.append(f"- Obfuscation: {extraction_result.obfuscation_layer}")
            if extraction_result.fallback_used:
                detail_parts.append(f"- Fallback technique: {extraction_result.technique_used}")
            if extraction_result.payload_text:
                snippet = extraction_result.payload_text[:500]
                if len(extraction_result.payload_text) > 500:
                    snippet += "... (truncated)"
                detail_parts.append(f"\n**Payload Preview:**\n```\n{snippet}\n```")

        if nlp_result and nlp_result.label != -1:
            detail_parts.append("\n#### Module 4 - NLP Classification")
            detail_parts.append(f"- **Payload type: {nlp_result.label_name}**")
            detail_parts.append(f"- Confidence: {nlp_result.confidence*100:.1f}%")
            detail_parts.append(f"- Classifier: {nlp_result.classifier_used}")
        elif extraction_result and extraction_result.error:
            detail_parts.append(f"\n#### Module 3 Error\n{extraction_result.error}")
        elif final_decision in ("lsb", "pvd"):
            detail_parts.append("\n#### Module 3\nExtraction quality: failed (no readable payload)")

        details_text = "\n".join(detail_parts)

        # Technical details
        tech_text = ""
        if show_technical:
            tech_text = (
                f"#### CNN Probabilities (Module 1)\n"
                f"| Class | Probability |\n|-------|-------------|\n"
                f"| clean | {prob_dict['clean']:.4f} |\n"
                f"| lsb | {prob_dict['lsb']:.4f} |\n"
                f"| pvd | {prob_dict['pvd']:.4f} |\n\n"
                f"#### Pipeline Info\n"
                f"- CNN: {class_name} ({conf*100:.1f}%) | Final: {final_decision} ({combined_conf*100:.1f}%)\n"
                f"- Override: {'Yes' if validation['override_occurred'] else 'No'}\n"
                f"- Model: {config.CNN_CONFIG['model_name']} ({config.CNN_CONFIG['input_size']}x{config.CNN_CONFIG['input_size']})\n"
                f"- Processing: {processing_time:.0f}ms | Device: {config.DEVICE} | Image: {w}x{h}\n"
            )

        # JSON report
        report = {
            "module1_cnn": {"prediction": class_name, "confidence": round(conf, 4),
                           "probabilities": {k: round(v, 4) for k, v in prob_dict.items()}},
            "module2_validation": {"final_decision": final_decision,
                                   "combined_confidence": round(combined_conf, 4),
                                   "override": validation["override_occurred"]},
            "module3_extraction": None,
            "module4_nlp": None,
            "risk_level": risk_label,
            "processing_time_ms": round(processing_time, 1),
        }
        if extraction_result:
            report["module3_extraction"] = {
                "quality": extraction_result.extraction_quality,
                "byte_length": extraction_result.payload_bytes_length,
                "obfuscation": extraction_result.obfuscation_layer,
            }
        if nlp_result and nlp_result.label != -1:
            report["module4_nlp"] = {
                "category": nlp_result.label_name,
                "confidence": round(nlp_result.confidence, 4),
                "classifier": nlp_result.classifier_used,
            }

        report_path = Path(tempfile.gettempdir()) / "stegodetect_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4, default=str)

        return detection_text, risk_text, "", details_text, tech_text, str(report_path)

    except Exception as e:
        return f"**Error:** {str(e)}", "", "", "", "", None


# ──────────────────────────────────────────────────────────────────────
# Batch analysis
# ──────────────────────────────────────────────────────────────────────
def analyze_batch(files):
    if not files:
        return "Upload images to analyze.", None
    if not PIPELINE_LOADED:
        return "[WARN] Models not loaded. Train first.", None

    results = []
    for file_obj in files:
        try:
            img = Image.open(file_obj.name).convert("RGB")
            temp_path = Path(tempfile.gettempdir()) / f"stegodetect_batch_{Path(file_obj.name).name}"
            img.save(str(temp_path), format="PNG")

            input_tensor = preprocess_image(img)
            pred_class, confidence, probs = MODEL.predict_with_confidence(input_tensor)
            class_name = config.CLASS_NAMES[pred_class.item()]
            conf = confidence.item()

            validation = validate_and_route(str(temp_path), cnn_prediction=class_name, cnn_confidence=conf)
            final_decision = validation["final_decision"]
            combined_conf = validation["combined_confidence"]

            payload_info = "-"
            nlp_info = "-"
            if final_decision in ("lsb", "pvd"):
                extraction = extract_payload(str(temp_path), final_decision, combined_conf)
                if extraction.extraction_quality != "failed" and extraction.payload_text:
                    nlp_result = NLP_CLASSIFIER.classify(extraction.payload_text, extraction.obfuscation_layer)
                    payload_info = f"{extraction.extraction_quality}"
                    nlp_info = nlp_result.label_name if nlp_result.label != -1 else "unknown"
                else:
                    payload_info = "failed"

            risk_label, _ = compute_risk(final_decision, nlp_info if nlp_info != "-" else None, combined_conf)

            results.append({
                "Filename": Path(file_obj.name).name,
                "Detection": final_decision.upper(),
                "Confidence": f"{combined_conf*100:.1f}%",
                "Payload": payload_info,
                "Category": nlp_info,
                "Risk": risk_label,
            })
        except Exception as e:
            results.append({
                "Filename": Path(file_obj.name).name,
                "Detection": "ERROR", "Confidence": "-",
                "Payload": "-", "Category": "-", "Risk": str(e)[:40],
            })

    clean_count = sum(1 for r in results if r["Detection"] == "CLEAN")
    stego_count = len(results) - clean_count
    summary = f"**{len(results)} images analyzed** | Clean: {clean_count} | Stego: {stego_count}"

    import pandas as pd
    return summary, pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────
# Premium CSS
# ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* Global */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* Header */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid rgba(99, 102, 241, 0.2);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15);
}
.header-banner h1 {
    color: #f8fafc !important;
    font-size: 2.2em !important;
    font-weight: 800 !important;
    margin: 0 0 4px 0 !important;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.header-banner p {
    color: #94a3b8 !important;
    font-size: 1.05em !important;
    margin: 0 !important;
}

/* Tabs */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95em !important;
    padding: 12px 24px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border-color: #6366f1 !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.45) !important;
}

/* Cards */
.result-card {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    padding: 16px !important;
}

/* Status indicator */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85em;
}
"""

# ──────────────────────────────────────────────────────────────────────
# About text
# ──────────────────────────────────────────────────────────────────────
ABOUT_TEXT = """
## What is Steganography?

**Steganography** is the practice of hiding secret data within ordinary, non-secret data
(such as images) to avoid detection. Unlike encryption, which makes data unreadable,
steganography hides the very existence of the secret message.

---

## StegoDetect Pipeline Architecture

StegoDetect is a **4-module AI-powered pipeline** for detecting, validating, extracting,
and classifying steganographic content in images.

| Module | Method | Purpose |
|--------|--------|---------|
| **Module 1** | EfficientNet-B4 CNN | Classify images as clean / LSB / PVD |
| **Module 2** | Chi-square + PPDH | Statistical confirmation or override |
| **Module 3** | LSB/PVD reversal | Extract hidden payload bytes |
| **Module 4** | Regex + TF-IDF + SVM | Classify payload type |

---

## Detection Techniques

### LSB (Least Significant Bit)
The most common image steganography technique. Data is hidden by replacing the least
significant bits of pixel values. Visually imperceptible, but detectable statistically.

### PVD (Pixel Value Differencing)
A more advanced technique that embeds data in the differences between adjacent pixel pairs.
Larger differences can hide more bits, making detection harder in textured regions.

---

## Payload Categories (Module 4)

| Category | Risk | Description |
|----------|------|-------------|
| PowerShell | Critical | Command execution scripts |
| JavaScript | High | Browser-based code injection |
| JavaScript_HTML | High | Embedded HTML with scripts |
| URL/IP | Medium-High | Command & control endpoints |
| Ethereum Address | Medium | Cryptocurrency wallet addresses |

---

## Technical Details

- **CNN Model:** EfficientNet-B4 (19M params, 380x380 input)
- **Framework:** PyTorch 2.1+ with mixed precision training
- **Statistical Tests:** Chi-square for LSB, PPDH for PVD
- **NLP Pipeline:** Regex -> TF-IDF+LinearSVC -> CodeBERT (optional)
- **Dataset:** ~60,000 images (512x512 PNG)

| Split | Clean | LSB | PVD | Total |
|-------|-------|-----|-----|-------|
| Train | 8,000 | 12,000 | 3,995 | 23,995 |
| Val | 4,000 | 6,000 | 2,000 | 12,000 |
| Test | 4,000 | 18,000 | 2,000 | 24,000 |
"""


# ──────────────────────────────────────────────────────────────────────
# Build UI
# ──────────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="StegoDetect - AI Steganography Detection",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # Header
        gr.HTML("""
        <div class="header-banner">
            <h1>StegoDetect</h1>
            <p>AI-Powered Steganography Detection & Payload Analysis Pipeline</p>
        </div>
        """)

        status = "Ready" if PIPELINE_LOADED else "Models not loaded - train first"
        status_color = "#10b981" if PIPELINE_LOADED else "#f59e0b"
        gr.HTML(f"""<div style="text-align:right;margin:-16px 0 12px 0;">
            <span style="color:{status_color};font-weight:600;font-size:0.85em;">
            {status}</span></div>""")

        # ── Tab 1: Analyze ───────────────────────────────────────────
        with gr.Tab("Analyze Image", id="analyze"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    image_input = gr.Image(label="Upload Image", type="pil", height=380)
                    with gr.Row():
                        show_tech = gr.Checkbox(label="Show technical details", value=False)
                        analyze_btn = gr.Button("Analyze Image", variant="primary",
                                                elem_classes=["primary-btn"], size="lg")

                with gr.Column(scale=3):
                    detection_output = gr.Markdown(label="Detection Result")
                    risk_output = gr.Markdown(label="Risk Level")
                    details_spacer = gr.Markdown(visible=False)
                    details_output = gr.Markdown(label="Analysis Details")
                    tech_output = gr.Markdown(label="Technical Details")
                    report_download = gr.File(label="Download JSON Report")

            analyze_btn.click(
                fn=analyze_single_image,
                inputs=[image_input, show_tech],
                outputs=[detection_output, risk_output, details_spacer,
                         details_output, tech_output, report_download],
            )

        # ── Tab 2: Batch ─────────────────────────────────────────────
        with gr.Tab("Batch Analysis", id="batch"):
            gr.Markdown("Upload multiple images for bulk analysis through the full 4-module pipeline.")
            file_input = gr.Files(label="Upload Images", file_types=["image"], file_count="multiple")
            batch_btn = gr.Button("Analyze Batch", variant="primary", elem_classes=["primary-btn"])
            batch_summary = gr.Markdown(label="Summary")
            batch_table = gr.Dataframe(
                label="Results",
                headers=["Filename", "Detection", "Confidence", "Payload", "Category", "Risk"],
            )
            batch_btn.click(fn=analyze_batch, inputs=[file_input], outputs=[batch_summary, batch_table])

        # ── Tab 3: About ─────────────────────────────────────────────
        with gr.Tab("About", id="about"):
            gr.Markdown(ABOUT_TEXT)

    return demo


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loaded = load_models()
    if not loaded:
        print("\n[WARN] Starting UI without models - analysis will show error messages.")
        print("   Train: python training/train_module1.py --epochs 50 --batch_size 16 --device cuda\n")

    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
