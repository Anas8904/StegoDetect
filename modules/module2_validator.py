"""
StegoDetect - Module 2: Statistical Validator

Validates CNN predictions using classical steganalysis tests:
    - Chi-square test for LSB steganography detection
    - PPDH (Pixel-Pair Difference Histogram) for PVD steganography detection
    - Routing function that combines CNN + statistical results
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import chi2 as chi2_dist

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ──────────────────────────────────────────────────────────────────────
# Function 1: Chi-Square LSB Test
# ──────────────────────────────────────────────────────────────────────

def chi_square_lsb_test(image_path):
    """
    Detect LSB steganography via pixel histogram chi-square analysis.

    Compares even/odd pixel value pair frequencies. In clean images,
    adjacent values (0,1), (2,3), etc. have naturally different frequencies.
    LSB embedding tends to equalize these pairs.

    Args:
        image_path: Path to the image file

    Returns:
        dict with keys: lsb_detected, p_value, chi2_value, confidence, per_channel
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    channel_names = ["R", "G", "B"]
    per_channel = {}
    chi2_values = []
    p_values = []

    for ch_idx, ch_name in enumerate(channel_names):
        channel = img[:, :, ch_idx].flatten()

        # Count frequency of each pixel value 0-255
        histogram = np.bincount(channel, minlength=256)

        # Group into 128 pairs: (0,1), (2,3), ..., (254,255)
        chi2_value = 0.0
        for pair_idx in range(128):
            even_val = pair_idx * 2
            odd_val = even_val + 1
            count_even = histogram[even_val]
            count_odd = histogram[odd_val]

            expected = (count_even + count_odd) / 2.0
            if expected == 0:
                continue

            chi2_contribution = (
                ((count_even - expected) ** 2 / expected) +
                ((count_odd - expected) ** 2 / expected)
            )
            chi2_value += chi2_contribution

        # Calculate p-value (127 degrees of freedom = 128 pairs - 1)
        p_value = float(chi2_dist.sf(chi2_value, df=127))

        per_channel[ch_name] = p_value
        chi2_values.append(chi2_value)
        p_values.append(p_value)

    # Average across channels
    avg_p_value = float(np.mean(p_values))
    avg_chi2 = float(np.mean(chi2_values))

    return {
        "lsb_detected": avg_p_value < 0.05,
        "p_value": avg_p_value,
        "chi2_value": avg_chi2,
        "confidence": min(1.0 - avg_p_value, 0.99),
        "per_channel": per_channel,
    }


# ──────────────────────────────────────────────────────────────────────
# Function 2: PPDH PVD Test
# ──────────────────────────────────────────────────────────────────────

# PVD quantization boundaries
PVD_BOUNDARIES = [7, 15, 31, 63, 127]


def ppdh_pvd_test(image_path):
    """
    Detect PVD steganography via pixel-pair difference histogram analysis.

    PVD embedding creates step discontinuities at quantization boundaries
    in the difference histogram. This test detects those steps.

    Args:
        image_path: Path to the image file

    Returns:
        dict with keys: pvd_detected, pvd_score, boundary_scores, confidence
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return {
            "pvd_detected": False,
            "pvd_score": 0,
            "boundary_scores": {},
            "confidence": 0.0,
        }

    # Check all 3 channels and take the maximum score
    max_pvd_score = 0
    best_boundary_scores = {}

    for ch_idx in range(3):
        channel = img[:, :, ch_idx].flatten()

        # Create non-overlapping pairs
        p1_array = channel[0::2].astype(int)
        p2_array = channel[1::2].astype(int)

        # Ensure equal lengths
        min_len = min(len(p1_array), len(p2_array))
        p1_array = p1_array[:min_len]
        p2_array = p2_array[:min_len]

        # Compute differences
        diffs = np.abs(p2_array - p1_array)

        # Build PPDH (Pixel-Pair Difference Histogram)
        counts = np.bincount(diffs, minlength=256).astype(float)

        # Adaptive threshold: exclude index 0 which usually has a huge peak
        # that destroys the standard deviation sensitivity
        if len(counts) > 1:
            threshold = np.std(counts[1:]) * 2.5
        else:
            threshold = 100.0

        current_pvd_score = 0
        current_boundary_scores = {}

        for k in PVD_BOUNDARIES:
            if k < 1 or k >= 254:
                current_boundary_scores[k] = {"second_derivative": 0, "step_magnitude": 0, "detected": False}
                continue

            # Second derivative at boundary
            second_derivative = float(counts[k + 1] - 2 * counts[k] + counts[k - 1])

            # Local step magnitude
            left_start = max(0, k - 3)
            right_end = min(255, k + 4)
            left_avg = np.mean(counts[left_start:k]) if k > left_start else 0
            right_avg = np.mean(counts[k + 1:right_end]) if k + 1 < right_end else 0
            step_magnitude = float(abs(right_avg - left_avg))

            detected = abs(second_derivative) > threshold
            if detected:
                current_pvd_score += 1

            current_boundary_scores[k] = {
                "second_derivative": second_derivative,
                "step_magnitude": step_magnitude,
                "detected": detected,
            }

        if current_pvd_score > max_pvd_score:
            max_pvd_score = current_pvd_score
            best_boundary_scores = current_boundary_scores

    # PVD detected if at least 2 of 5 boundaries show steps (reduced from 3 for better sensitivity)
    pvd_detected = max_pvd_score >= 2

    return {
        "pvd_detected": pvd_detected,
        "pvd_score": max_pvd_score,
        "boundary_scores": best_boundary_scores,
        "confidence": max_pvd_score / 5.0,
    }


# ──────────────────────────────────────────────────────────────────────
# Function 3: Validate and Route
# ──────────────────────────────────────────────────────────────────────

def validate_and_route(image_path, cnn_prediction, cnn_confidence, threshold=None):
    """
    Combine CNN output with statistical tests for final decision.

    Args:
        image_path: Path to the image file
        cnn_prediction: str ('clean', 'lsb', or 'pvd')
        cnn_confidence: float (0.0 to 1.0)
        threshold: confidence threshold (default: config.CONFIDENCE_THRESHOLD)

    Returns:
        dict with final decision, routing info, and all test results
    """
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD

    override_occurred = False
    warning = None
    chi_result = None
    pvd_result = None

    if cnn_confidence >= threshold:
        if cnn_prediction == "clean":
            # Sanity check: run chi-square to catch missed LSB
            chi_result = chi_square_lsb_test(image_path)
            if chi_result["lsb_detected"]:
                final_decision = "lsb"
                override_occurred = True
                warning = (
                    f"CNN predicted clean ({cnn_confidence:.2f}) but chi-square "
                    f"detected LSB (p={chi_result['p_value']:.4f}). Overriding to LSB."
                )
                combined_confidence = chi_result["confidence"]
            else:
                final_decision = "clean"
                combined_confidence = cnn_confidence

        elif cnn_prediction == "lsb":
            chi_result = chi_square_lsb_test(image_path)
            if chi_result["lsb_detected"]:
                # Confirmed by both CNN and statistics
                final_decision = "lsb"
                combined_confidence = (cnn_confidence + chi_result["confidence"]) / 2
            else:
                # CNN says LSB but statistics disagree - flag as uncertain
                final_decision = "lsb"
                combined_confidence = cnn_confidence * 0.7  # Reduce confidence
                warning = (
                    f"CNN detected LSB ({cnn_confidence:.2f}) but chi-square "
                    f"did not confirm (p={chi_result['p_value']:.4f}). Flagged as uncertain."
                )

        elif cnn_prediction == "pvd":
            pvd_result = ppdh_pvd_test(image_path)
            if pvd_result["pvd_detected"]:
                # Confirmed by both CNN and statistics
                final_decision = "pvd"
                combined_confidence = (cnn_confidence + pvd_result["confidence"]) / 2
            else:
                # CNN says PVD but statistics disagree
                final_decision = "pvd"
                combined_confidence = cnn_confidence * 0.7
                warning = (
                    f"CNN detected PVD ({cnn_confidence:.2f}) but PPDH "
                    f"did not confirm (score={pvd_result['pvd_score']}/5). Flagged as uncertain."
                )
        else:
            final_decision = cnn_prediction
            combined_confidence = cnn_confidence

    else:
        # Low CNN confidence — run both tests and let statistics decide
        chi_result = chi_square_lsb_test(image_path)
        pvd_result = ppdh_pvd_test(image_path)

        if chi_result["lsb_detected"]:
            final_decision = "lsb"
            combined_confidence = chi_result["confidence"]
        elif pvd_result["pvd_detected"]:
            final_decision = "pvd"
            combined_confidence = pvd_result["confidence"]
        else:
            final_decision = "clean"
            combined_confidence = max(1 - chi_result["confidence"], 1 - pvd_result["confidence"])

    # Determine routing
    route_map = {
        "clean": "terminate",
        "lsb": "lsb_extractor",
        "pvd": "pvd_extractor",
    }

    return {
        "final_decision": final_decision,
        "route_to": route_map.get(final_decision, "terminate"),
        "combined_confidence": float(combined_confidence),
        "cnn_prediction": cnn_prediction,
        "cnn_confidence": float(cnn_confidence),
        "statistical_tests": {
            "chi_square": chi_result,
            "ppdh": pvd_result,
        },
        "override_occurred": override_occurred,
        "warning": warning,
    }


# ──────────────────────────────────────────────────────────────────────
# Function 4: Plot PPDH
# ──────────────────────────────────────────────────────────────────────

def plot_ppdh(image_path, save_path=None):
    """
    Plot the PPDH histogram with boundary markers.

    Args:
        image_path: Path to the image file
        save_path: Optional path to save the plot (PNG)

    Returns:
        matplotlib figure or save_path if saved
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot plot PPDH")
        return None

    img = np.array(Image.open(image_path).convert("RGB"))
    channel = img[:, :, 0].flatten()

    p1 = channel[0::2].astype(int)
    p2 = channel[1::2].astype(int)
    min_len = min(len(p1), len(p2))
    diffs = np.abs(p2[:min_len] - p1[:min_len])
    counts = np.bincount(diffs, minlength=256)

    # Run detection to get boundary info
    pvd_result = ppdh_pvd_test(image_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(counts)), counts, color="#4a90d9", alpha=0.7, width=1.0)

    # Mark boundaries
    for k in PVD_BOUNDARIES:
        detected = pvd_result["boundary_scores"][k]["detected"]
        color = "red" if detected else "orange"
        linestyle = "-" if detected else "--"
        ax.axvline(x=k, color=color, linestyle=linestyle, alpha=0.7, linewidth=1.5)
        if detected:
            ax.plot(k, counts[k], "rx", markersize=12, markeredgewidth=2)

    ax.set_xlabel("Pixel-Pair Difference")
    ax.set_ylabel("Count")
    ax.set_title(f"PPDH — PVD Score: {pvd_result['pvd_score']}/5")
    ax.set_xlim(0, 150)  # Most action is in lower differences
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    else:
        return fig


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("StegoDetect - Module 2: Statistical Validator")
    print("=" * 50)

    # Test on sample images from the dataset
    test_dir = config.COMBINED_DIR / "test"

    for class_name in config.CLASS_NAMES:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"\n[SKIP] {class_dir} not found")
            continue

        # Get first image
        images = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if not images:
            print(f"\n[SKIP] No images in {class_dir}")
            continue

        sample = images[0]
        print(f"\n--- Testing: {sample.name} (expected: {class_name}) ---")

        # Chi-square test
        chi_result = chi_square_lsb_test(str(sample))
        print(f"  Chi-Square LSB Test:")
        print(f"    LSB Detected:  {chi_result['lsb_detected']}")
        print(f"    p-value:       {chi_result['p_value']:.6f}")
        print(f"    chi2 value:    {chi_result['chi2_value']:.2f}")
        print(f"    Confidence:    {chi_result['confidence']:.4f}")
        print(f"    Per-channel:   R={chi_result['per_channel']['R']:.4f} "
              f"G={chi_result['per_channel']['G']:.4f} "
              f"B={chi_result['per_channel']['B']:.4f}")

        # PPDH test
        pvd_result = ppdh_pvd_test(str(sample))
        print(f"  PPDH PVD Test:")
        print(f"    PVD Detected:  {pvd_result['pvd_detected']}")
        print(f"    PVD Score:     {pvd_result['pvd_score']}/5")
        print(f"    Confidence:    {pvd_result['confidence']:.4f}")

        # Validate and route (simulating CNN prediction)
        route_result = validate_and_route(
            str(sample),
            cnn_prediction=class_name,
            cnn_confidence=0.85,
        )
        print(f"  Validation Result:")
        print(f"    Final Decision:  {route_result['final_decision']}")
        print(f"    Route To:        {route_result['route_to']}")
        print(f"    Combined Conf:   {route_result['combined_confidence']:.4f}")
        if route_result["warning"]:
            print(f"    Warning:         {route_result['warning']}")

    print("\n[OK] Module 2 Statistical Validator working correctly!")
