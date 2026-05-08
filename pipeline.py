"""
StegoDetect - End-to-End Inference Pipeline

Chains all 4 modules into a single analysis pipeline:
    Module 1 (CNN)        -> Steganography detection
    Module 2 (Statistics) -> Validation & routing
    Module 3 (Extraction) -> Payload recovery
    Module 4 (NLP)        -> Payload classification

Usage:
    python pipeline.py --image path/to/image.png
    python pipeline.py --folder path/to/folder/ --output ./results/
    python pipeline.py --test
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from modules.module1_cnn import StegoClassifier, load_pretrained_stego
from modules.dataset import get_transforms
from modules.module2_validator import validate_and_route
from modules.module3_extractor import extract_payload
from modules.module4_nlp import PayloadClassifier, NLPResult, LABEL_NAMES


# ──────────────────────────────────────────────────────────────────────
# Dataclass: StegoReport
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StegoReport:
    """Complete analysis report for a single image."""
    image_path: str = ""
    image_filename: str = ""
    technique_detected: str = "clean"       # clean, lsb, pvd, uncertain
    detection_confidence: float = 0.0
    statistical_confirmation: bool = False
    override_occurred: bool = False
    extraction_attempted: bool = False
    extraction_success: bool = False
    extraction_quality: str = "failed"
    obfuscation_layer: str = "none"
    payload_snippet: str = ""               # first 300 chars
    payload_full_length: int = 0
    payload_category: Optional[str] = None
    category_confidence: Optional[float] = None
    classifier_used: Optional[str] = None
    risk_level: str = "none"
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# Risk Level Computation
# ──────────────────────────────────────────────────────────────────────

RISK_ORDER = ["none", "low", "medium", "high", "critical"]

def compute_risk_level(technique, payload_category, detection_confidence):
    """
    Compute risk level based on technique and payload type.

    Risk matrix:
        clean            -> none
        uncertain        -> low
        lsb/pvd + PS     -> critical
        lsb/pvd + JS     -> high
        lsb/pvd + HTML   -> high
        lsb/pvd + URL    -> medium/high
        lsb/pvd + ETH    -> medium
        lsb/pvd + None   -> medium
    """
    if technique == "clean":
        return "none"
    if technique == "uncertain":
        return "low"

    # Stego detected
    risk = "medium"  # default for failed extraction
    if payload_category:
        risk_map = {
            "PowerShell": "critical",
            "JavaScript": "high",
            "JavaScript_HTML": "high",
            "URL_IP": "high" if technique == "pvd" else "medium",
            "Ethereum_Address": "medium",
        }
        risk = risk_map.get(payload_category, "medium")

    # Downgrade by one level if low confidence
    if detection_confidence < 0.70:
        idx = RISK_ORDER.index(risk)
        if idx > 0:
            risk = RISK_ORDER[idx - 1]

    return risk


# ──────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────

class StegoDetectPipeline:
    """
    End-to-end steganography detection and analysis pipeline.

    Chains Module 1 (CNN) -> Module 2 (Stats) -> Module 3 (Extraction) -> Module 4 (NLP)
    """

    def __init__(
        self,
        module1_checkpoint=None,
        models_dir=None,
        device="auto",
        use_codebert=False,
        confidence_threshold=None,
    ):
        if device == "auto":
            self.device = config.DEVICE
        else:
            self.device = torch.device(device)

        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD

        # Module 1: CNN
        self.model = None
        self.transform = get_transforms("test")
        ckpt_path = module1_checkpoint or str(config.MODULE1_CHECKPOINT_DIR / "best_model.pth")
        if os.path.exists(ckpt_path):
            self.model = load_pretrained_stego(ckpt_path, device=self.device)
            print("[OK] Module 1 (CNN) loaded")
        else:
            print(f"[WARN] Module 1 checkpoint not found: {ckpt_path}")

        # Module 4: NLP
        self.nlp = PayloadClassifier(use_codebert=use_codebert)
        models_path = models_dir or str(config.MODELS_DIR)
        nlp_loaded = self.nlp.load_all(models_path)
        if nlp_loaded:
            print("[OK] Module 4 (NLP) loaded")
        else:
            print("[INFO] Module 4 using keyword fallback (models not trained yet)")

        # Modules 2 & 3 are stateless functions
        print("[OK] Modules 2 (Stats) and 3 (Extraction) ready")

    def _preprocess_image(self, pil_image):
        """Preprocess a PIL image for CNN inference."""
        import cv2
        image_tensor = self.transform(pil_image)

        if getattr(self.model, "use_noise_residual", False):
            from modules.dataset import StegoDataset
            size = config.CNN_CONFIG["input_size"]
            img_resized = pil_image.resize((size, size), Image.BILINEAR)
            img_np = np.array(img_resized, dtype=np.float64)
            residuals = []
            for kernel in [StegoDataset._SRM_KERNEL_1, StegoDataset._SRM_KERNEL_2, StegoDataset._SRM_KERNEL_3]:
                ch_res = []
                for c in range(3):
                    filtered = cv2.filter2D(img_np[:, :, c], cv2.CV_64F, kernel)
                    ch_res.append(filtered)
                residuals.append(np.mean(ch_res, axis=0))
            noise_map = np.clip(np.mean(residuals, axis=0) / 4.0, -3.0, 3.0)
            noise_channel = torch.tensor(noise_map, dtype=torch.float32).unsqueeze(0)
            image_tensor = torch.cat([image_tensor, noise_channel], dim=0)

        return image_tensor.unsqueeze(0).to(self.device)

    def analyze(self, image_path):
        """
        Run complete 4-module pipeline on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            StegoReport dataclass
        """
        report = StegoReport(
            image_path=str(image_path),
            image_filename=Path(image_path).name,
        )

        start_time = time.time()

        try:
            # ── Validate image ────────────────────────────────────────
            pil_image = Image.open(image_path).convert("RGB")
            w, h = pil_image.size
            if w < 50 or h < 50:
                report.error = f"Image too small: {w}x{h}"
                return report

            # ── STEP 1: Module 1 — CNN Detection ─────────────────────
            if self.model is None:
                report.error = "Module 1 model not loaded"
                return report

            input_tensor = self._preprocess_image(pil_image)
            pred_class, confidence, probs = self.model.predict_with_confidence(input_tensor)

            pred_idx = pred_class.item()
            conf = confidence.item()
            class_name = config.CLASS_NAMES[pred_idx]

            # ── STEP 2: Module 2 — Statistical Validation ────────────
            validation = validate_and_route(
                str(image_path),
                cnn_prediction=class_name,
                cnn_confidence=conf,
                threshold=self.confidence_threshold,
            )

            final_decision = validation["final_decision"]
            combined_conf = validation["combined_confidence"]

            report.technique_detected = final_decision
            report.detection_confidence = combined_conf
            report.override_occurred = validation["override_occurred"]

            if validation["warning"]:
                report.warnings.append(validation["warning"])

            # Check statistical confirmation
            chi_result = validation["statistical_tests"]["chi_square"]
            pvd_result = validation["statistical_tests"]["ppdh"]
            if final_decision == "lsb" and chi_result and chi_result["lsb_detected"]:
                report.statistical_confirmation = True
            elif final_decision == "pvd" and pvd_result and pvd_result["pvd_detected"]:
                report.statistical_confirmation = True
            elif final_decision == "clean":
                report.statistical_confirmation = True

            # ── STEP 3: Module 3 — Payload Extraction ────────────────
            if final_decision in ("lsb", "pvd"):
                report.extraction_attempted = True
                extraction = extract_payload(
                    str(image_path),
                    technique=final_decision,
                    confidence=combined_conf,
                )

                report.extraction_quality = extraction.extraction_quality
                report.obfuscation_layer = extraction.obfuscation_layer

                if extraction.extraction_quality != "failed" and extraction.payload_text:
                    report.extraction_success = True
                    report.payload_full_length = extraction.payload_bytes_length
                    report.payload_snippet = extraction.payload_text[:300]

                    # ── STEP 4: Module 4 — NLP Classification ────────
                    nlp_result = self.nlp.classify(
                        extraction.payload_text,
                        obfuscation_type=extraction.obfuscation_layer,
                    )
                    report.payload_category = nlp_result.label_name
                    report.category_confidence = nlp_result.confidence
                    report.classifier_used = nlp_result.classifier_used

                if extraction.error:
                    report.warnings.append(f"Extraction: {extraction.error}")

            # ── STEP 5: Risk Level ────────────────────────────────────
            report.risk_level = compute_risk_level(
                report.technique_detected,
                report.payload_category,
                report.detection_confidence,
            )

        except Exception as e:
            report.error = str(e)

        report.processing_time_ms = (time.time() - start_time) * 1000
        return report

    def analyze_batch(self, image_paths, max_workers=4):
        """
        Process multiple images in parallel.

        Returns:
            (list of StegoReport, summary dict)
        """
        reports = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.analyze, path): path
                for path in image_paths
            }
            for future in as_completed(futures):
                try:
                    report = future.result()
                    reports.append(report)
                except Exception as e:
                    path = futures[future]
                    r = StegoReport(
                        image_path=str(path),
                        image_filename=Path(path).name,
                        error=str(e),
                    )
                    reports.append(r)

        # Summary
        summary = {
            "total": len(reports),
            "clean": sum(1 for r in reports if r.technique_detected == "clean"),
            "lsb_detected": sum(1 for r in reports if r.technique_detected == "lsb"),
            "pvd_detected": sum(1 for r in reports if r.technique_detected == "pvd"),
            "critical": sum(1 for r in reports if r.risk_level == "critical"),
            "high": sum(1 for r in reports if r.risk_level == "high"),
            "medium": sum(1 for r in reports if r.risk_level == "medium"),
            "failed": sum(1 for r in reports if r.error is not None),
        }

        return reports, summary


# ──────────────────────────────────────────────────────────────────────
# Formatting & Serialization
# ──────────────────────────────────────────────────────────────────────

def format_report(report: StegoReport) -> str:
    """Format a StegoReport as a readable string."""
    lines = [
        f"{'='*60}",
        f"  StegoDetect Analysis Report",
        f"{'='*60}",
        f"  Image:      {report.image_filename}",
        f"  Technique:  {report.technique_detected.upper()}",
        f"  Confidence: {report.detection_confidence*100:.1f}%",
        f"  Risk Level: {report.risk_level.upper()}",
        f"  Stats OK:   {'Yes' if report.statistical_confirmation else 'No'}",
    ]

    if report.extraction_attempted:
        lines.append(f"  Extracted:  {'Yes' if report.extraction_success else 'No'}")
        lines.append(f"  Quality:    {report.extraction_quality}")
        if report.obfuscation_layer != "none":
            lines.append(f"  Obfuscation: {report.obfuscation_layer}")

    if report.payload_category:
        lines.append(f"  Payload:    {report.payload_category} ({report.category_confidence*100:.1f}%)")
        lines.append(f"  Classifier: {report.classifier_used}")

    if report.payload_snippet:
        snippet = report.payload_snippet[:100]
        if len(report.payload_snippet) > 100:
            snippet += "..."
        lines.append(f"  Preview:    {snippet}")

    lines.append(f"  Time:       {report.processing_time_ms:.0f} ms")

    if report.warnings:
        for w in report.warnings:
            lines.append(f"  [!] {w}")
    if report.error:
        lines.append(f"  [ERROR] {report.error}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def report_to_dict(report: StegoReport) -> dict:
    """Convert StegoReport to a JSON-serializable dict."""
    return asdict(report)


def save_report(report: StegoReport, output_dir=None):
    """Save report as JSON file."""
    if output_dir is None:
        output_dir = str(config.REPORTS_DIR)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(report.image_filename).stem
    filename = f"{stem}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(report_to_dict(report), f, indent=4, default=str)

    return filepath


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="StegoDetect Pipeline")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--output", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--test", action="store_true", help="Run on test set samples")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-codebert", action="store_true", help="Disable CodeBERT")
    args = parser.parse_args()

    pipeline = StegoDetectPipeline(
        device=args.device,
        use_codebert=not args.no_codebert,
    )

    if args.image:
        report = pipeline.analyze(args.image)
        print(format_report(report))
        if args.output:
            path = save_report(report, args.output)
            print(f"\nReport saved to {path}")

    elif args.folder:
        folder = Path(args.folder)
        images = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])
        print(f"Found {len(images)} images in {folder}")

        reports, summary = pipeline.analyze_batch([str(p) for p in images])
        for r in reports:
            print(format_report(r))

        print(f"\n--- Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        if args.output:
            for r in reports:
                save_report(r, args.output)
            print(f"\nReports saved to {args.output}")

    elif args.test:
        print("\n--- Running test on sample images ---")
        test_dir = config.COMBINED_DIR / "test"
        for class_name in config.CLASS_NAMES:
            class_dir = test_dir / class_name
            if not class_dir.exists():
                continue
            images = sorted([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ])[:2]  # 2 samples per class
            for img in images:
                report = pipeline.analyze(str(img))
                print(format_report(report))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
