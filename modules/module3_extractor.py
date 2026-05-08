"""
StegoDetect - Module 3: Payload Extractor

Extracts hidden payloads from confirmed stego images.
Pure algorithmic code — NO machine learning.

Supports:
    - LSB (Least Significant Bit) extraction
    - PVD (Pixel Value Differencing) extraction
    - Obfuscation detection and reversal (base64, zip)
    - Extraction quality assessment
"""

import sys
import re
import base64
import zipfile
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# PVD quantization table: (lower, upper, bits_to_encode)
PVD_RANGES = [
    (0, 7, 3),
    (8, 15, 3),
    (16, 31, 4),
    (32, 63, 5),
    (64, 127, 6),
    (128, 255, 7),
]


# ──────────────────────────────────────────────────────────────────────
# Custom Exception
# ──────────────────────────────────────────────────────────────────────

class ExtractionError(Exception):
    """Raised when payload extraction fails."""
    pass


# ──────────────────────────────────────────────────────────────────────
# Dataclass for Extraction Results
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Result of a payload extraction operation."""
    payload_text: str = ""
    payload_bytes_length: int = 0
    obfuscation_layer: str = "none"
    extraction_quality: str = "failed"
    printable_ratio: float = 0.0
    technique_used: str = ""
    fallback_used: bool = False
    likely_language: Optional[str] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# Function 1: LSB Extraction
# ──────────────────────────────────────────────────────────────────────

def extract_lsb(image_path, bits_per_channel=1):
    """
    Extract hidden data from LSB steganography.

    Args:
        image_path: Path to the stego image
        bits_per_channel: Number of LSBs to extract per channel (1, 2, or 3)

    Returns:
        bytes: The extracted payload bytes

    Raises:
        ExtractionError: If the image cannot be loaded
        ValueError: If bits_per_channel is not in [1, 2, 3]
    """
    if bits_per_channel not in [1, 2, 3]:
        raise ValueError(f"bits_per_channel must be 1, 2, or 3, got {bits_per_channel}")

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ExtractionError(f"Cannot open image: {image_path}")

    # Extract LSBs
    mask = (2 ** bits_per_channel) - 1
    lsb_array = img & mask

    # Flatten in row-major order (B, G, R for cv2)
    bits_raw = lsb_array.flatten()

    if bits_per_channel == 1:
        bits = bits_raw.astype(np.uint8)
    else:
        all_bits = []
        for val in bits_raw:
            for bit_pos in range(bits_per_channel - 1, -1, -1):
                all_bits.append((int(val) >> bit_pos) & 1)
        bits = np.array(all_bits, dtype=np.uint8)

    # Pack bits into bytes
    n_usable = len(bits) - (len(bits) % 8)
    byte_array = np.packbits(bits[:n_usable])

    # Dynamic Length Header Parsing
    max_len = len(byte_array)
    payload_bytes = byte_array

    # 8-byte header
    len8 = int.from_bytes(byte_array[:8], 'big')
    if 0 < len8 <= max_len - 8:
        payload_bytes = byte_array[8:8+len8]
    else:
        # 4-byte header
        len4 = int.from_bytes(byte_array[:4], 'big')
        if 0 < len4 <= max_len - 4:
            payload_bytes = byte_array[4:4+len4]
        else:
            # 2-byte header
            len2 = int.from_bytes(byte_array[:2], 'big')
            if 0 < len2 <= max_len - 2:
                payload_bytes = byte_array[2:2+len2]
            else:
                # No valid header found, return up to first null byte
                null_positions = np.where(byte_array == 0)[0]
                if len(null_positions) > 0:
                    payload_bytes = byte_array[:null_positions[0]]

    return bytes(payload_bytes)


# ──────────────────────────────────────────────────────────────────────
# Function 2: PVD Extraction
# ──────────────────────────────────────────────────────────────────────

def _get_pvd_range(diff):
    """Find which PVD quantization range a difference value falls into."""
    for lower, upper, n_bits in PVD_RANGES:
        if lower <= diff <= upper:
            return lower, upper, n_bits
    return None, None, None


def extract_pvd(image_path):
    """
    Extract hidden data from PVD steganography.

    Uses the red channel and non-overlapping pixel pairs.

    Args:
        image_path: Path to the stego image

    Returns:
        bytes: The extracted payload bytes

    Raises:
        ExtractionError: If the image cannot be loaded
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ExtractionError(f"Cannot open image: {image_path}")

    # Use Red channel (index 2 in BGR)
    channel = img[:, :, 2].flatten().astype(int)

    # Create non-overlapping pairs
    p1 = channel[0::2]
    p2 = channel[1::2]
    min_len = min(len(p1), len(p2))
    p1, p2 = p1[:min_len], p2[:min_len]

    # Compute differences
    diffs = np.abs(p2 - p1)

    # Decode bits from each pair
    recovered_bits = []
    for i in range(min_len):
        d = int(diffs[i])
        range_lower, range_upper, n_bits = _get_pvd_range(d)

        if range_lower is None:
            continue

        secret_value = d - range_lower

        # Clamp to valid range for n_bits
        max_val = (2 ** n_bits) - 1
        secret_value = min(secret_value, max_val)

        # Convert to binary digits
        bits_str = format(secret_value, f"0{n_bits}b")
        recovered_bits.extend([int(b) for b in bits_str])

    # Pack bits into bytes
    n_usable = len(recovered_bits) - (len(recovered_bits) % 8)
    recovered_bits = recovered_bits[:n_usable]

    byte_list = []
    for i in range(0, len(recovered_bits), 8):
        byte_val = int("".join(map(str, recovered_bits[i:i + 8])), 2)
        byte_list.append(byte_val)

    byte_array = bytes(byte_list)
    
    # Strip leading zeros that might come from uniform background pixels
    byte_array = byte_array.lstrip(b"\x00")

    # Dynamic Length Header Parsing
    max_len = len(byte_array)
    payload_bytes = byte_array

    if max_len >= 8:
        # 8-byte header
        len8 = int.from_bytes(byte_array[:8], 'big')
        if 0 < len8 <= max_len - 8:
            payload_bytes = byte_array[8:8+len8]
        else:
            # 4-byte header
            len4 = int.from_bytes(byte_array[:4], 'big')
            if 0 < len4 <= max_len - 4:
                payload_bytes = byte_array[4:4+len4]
            else:
                # 2-byte header
                len2 = int.from_bytes(byte_array[:2], 'big')
                if 0 < len2 <= max_len - 2:
                    payload_bytes = byte_array[2:2+len2]
                else:
                    # No valid header found, return up to first null byte
                    null_pos = byte_array.find(b"\x00")
                    if null_pos >= 0:
                        payload_bytes = byte_array[:null_pos]
    else:
        null_pos = byte_array.find(b"\x00")
        if null_pos >= 0:
            payload_bytes = byte_array[:null_pos]

    return payload_bytes


# ──────────────────────────────────────────────────────────────────────
# Function 3: Obfuscation Detection
# ──────────────────────────────────────────────────────────────────────

def detect_obfuscation(raw_bytes):
    """
    Detect the obfuscation type of raw payload bytes.

    Returns:
        str: 'zip', 'base64', 'none', or 'unknown'
    """
    if not raw_bytes or len(raw_bytes) < 4:
        return "none"

    # Check ZIP magic bytes
    if raw_bytes[:4] == b"PK\x03\x04":
        return "zip"

    # Check Base64
    try:
        text = raw_bytes.decode("ascii", errors="strict")
        # Check if all characters are in base64 alphabet
        b64_pattern = re.compile(r'^[A-Za-z0-9+/=\n\r\s]+$')
        if b64_pattern.match(text) and len(text) >= 4:
            # Try actually decoding
            try:
                decoded = base64.b64decode(text, validate=True)
                if len(decoded) > 0:
                    return "base64"
            except Exception:
                # Try without strict validation (padding might be off)
                try:
                    decoded = base64.b64decode(text + "==")
                    if len(decoded) > 0:
                        return "base64"
                except Exception:
                    pass
    except (UnicodeDecodeError, ValueError):
        pass

    return "none"


# ──────────────────────────────────────────────────────────────────────
# Function 4: Deobfuscation
# ──────────────────────────────────────────────────────────────────────

def deobfuscate(raw_bytes, obfuscation_type=None, max_depth=3):
    """
    Recursively deobfuscate payload bytes.

    Supports nested obfuscation (e.g., base64-encoded zip).

    Args:
        raw_bytes: The raw payload bytes
        obfuscation_type: Type of obfuscation (auto-detected if None)
        max_depth: Maximum recursion depth for nested obfuscation

    Returns:
        tuple: (deobfuscated_bytes, obfuscation_layers_list)
    """
    if max_depth <= 0 or not raw_bytes:
        return raw_bytes, []

    if obfuscation_type is None:
        obfuscation_type = detect_obfuscation(raw_bytes)

    layers = []

    if obfuscation_type == "zip":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                # Extract first file in the archive
                names = zf.namelist()
                if names:
                    decompressed = zf.read(names[0])
                    layers.append("zip")
                    # Check for nested obfuscation
                    result, nested_layers = deobfuscate(decompressed, max_depth=max_depth - 1)
                    layers.extend(nested_layers)
                    return result, layers
        except (zipfile.BadZipFile, Exception):
            return raw_bytes, []

    elif obfuscation_type == "base64":
        try:
            text = raw_bytes.decode("ascii", errors="ignore").strip()
            decoded = base64.b64decode(text, validate=False)
            layers.append("base64")
            # Check for nested obfuscation
            result, nested_layers = deobfuscate(decoded, max_depth=max_depth - 1)
            layers.extend(nested_layers)
            return result, layers
        except Exception:
            return raw_bytes, []

    return raw_bytes, layers


# ──────────────────────────────────────────────────────────────────────
# Function 5: Extraction Quality Assessment
# ──────────────────────────────────────────────────────────────────────

def assess_extraction_quality(raw_bytes):
    """
    Assess the quality of extracted payload bytes.

    Returns:
        dict with keys: quality, printable_ratio, is_valid_utf8,
                        likely_language, byte_length
    """
    if not raw_bytes:
        return {
            "quality": "failed",
            "printable_ratio": 0.0,
            "is_valid_utf8": False,
            "likely_language": None,
            "byte_length": 0,
        }

    byte_length = len(raw_bytes)

    # Try to decode as UTF-8
    is_valid_utf8 = False
    try:
        text = raw_bytes.decode("utf-8")
        is_valid_utf8 = True
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="replace")

    # Calculate printable ratio
    if len(text) > 0:
        printable_count = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
        printable_ratio = printable_count / len(text)
    else:
        printable_ratio = 0.0

    # Determine quality
    if printable_ratio > 0.85:
        quality = "good"
    elif printable_ratio >= 0.50:
        quality = "partial"
    else:
        quality = "failed"

    # Quick language detection via keywords
    likely_language = None
    text_lower = text.lower() if text else ""

    if re.search(r"0x[0-9a-fA-F]{40}", text):
        likely_language = "ethereum"
    elif any(kw in text for kw in ["Invoke-", "$env:", "Get-", "Set-", "New-Object", "-EncodedCommand"]):
        likely_language = "powershell"
    elif any(kw in text_lower for kw in ["<script", "<html", "<iframe", "onclick="]):
        likely_language = "html"
    elif any(kw in text for kw in ["function", "eval(", "document.", "=>", "var ", "const ", "let "]):
        likely_language = "javascript"
    elif "http" in text_lower:
        likely_language = "url"

    return {
        "quality": quality,
        "printable_ratio": printable_ratio,
        "is_valid_utf8": is_valid_utf8,
        "likely_language": likely_language,
        "byte_length": byte_length,
    }


# ──────────────────────────────────────────────────────────────────────
# Function 6: Main Extraction Function
# ──────────────────────────────────────────────────────────────────────

def extract_payload(image_path, technique, confidence=1.0):
    """
    Main extraction function that orchestrates the full extraction pipeline.

    Args:
        image_path: Path to the stego image
        technique: 'lsb' or 'pvd'
        confidence: CNN confidence (used to decide if fallback is tried)

    Returns:
        ExtractionResult dataclass
    """
    result = ExtractionResult(technique_used=technique)

    try:
        # Primary extraction
        if technique == "lsb":
            raw_bytes = extract_lsb(image_path)
        elif technique == "pvd":
            raw_bytes = extract_pvd(image_path)
        else:
            result.error = f"Unknown technique: {technique}"
            return result

        # Assess quality of raw extraction
        quality_info = assess_extraction_quality(raw_bytes)

        # If quality is poor and confidence is low, try the other technique
        if quality_info["quality"] == "failed" and confidence < 0.80:
            fallback_technique = "pvd" if technique == "lsb" else "lsb"
            try:
                if fallback_technique == "lsb":
                    fallback_bytes = extract_lsb(image_path)
                else:
                    fallback_bytes = extract_pvd(image_path)

                fallback_quality = assess_extraction_quality(fallback_bytes)
                if fallback_quality["printable_ratio"] > quality_info["printable_ratio"]:
                    raw_bytes = fallback_bytes
                    quality_info = fallback_quality
                    result.fallback_used = True
                    result.technique_used = fallback_technique
            except Exception:
                pass  # Fallback failed, stick with original

        # Deobfuscate
        obfuscation_type = detect_obfuscation(raw_bytes)
        if obfuscation_type != "none":
            deobfuscated, layers = deobfuscate(raw_bytes, obfuscation_type)
            result.obfuscation_layer = " > ".join(layers) if layers else "none"

            # Re-assess quality after deobfuscation
            final_quality = assess_extraction_quality(deobfuscated)
            if final_quality["printable_ratio"] >= quality_info["printable_ratio"]:
                raw_bytes = deobfuscated
                quality_info = final_quality

        # Build final result
        try:
            result.payload_text = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            result.payload_text = raw_bytes.decode("latin-1", errors="replace")

        result.payload_bytes_length = len(raw_bytes)
        result.extraction_quality = quality_info["quality"]
        result.printable_ratio = quality_info["printable_ratio"]
        result.likely_language = quality_info["likely_language"]

    except ExtractionError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"Unexpected error: {str(e)}"

    return result


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("StegoDetect - Module 3: Payload Extractor")
    print("=" * 50)

    test_dir = config.COMBINED_DIR / "test"

    # Test LSB extraction
    lsb_dir = test_dir / "lsb"
    if lsb_dir.exists():
        images = sorted([f for f in lsb_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if images:
            sample = images[0]
            print(f"\n--- LSB Extraction: {sample.name} ---")
            result = extract_payload(str(sample), "lsb")
            print(f"  Quality:     {result.extraction_quality}")
            print(f"  Printable:   {result.printable_ratio:.2%}")
            print(f"  Byte Length: {result.payload_bytes_length}")
            print(f"  Obfuscation: {result.obfuscation_layer}")
            print(f"  Language:    {result.likely_language}")
            print(f"  Fallback:    {result.fallback_used}")
            if result.error:
                print(f"  Error:       {result.error}")
            snippet = result.payload_text[:200] if result.payload_text else "(empty)"
            print(f"  Payload (first 200 chars):\n    {snippet}")
        else:
            print("[SKIP] No LSB test images found")
    else:
        print(f"[SKIP] {lsb_dir} not found")

    # Test PVD extraction
    pvd_dir = test_dir / "pvd"
    if pvd_dir.exists():
        images = sorted([f for f in pvd_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if images:
            sample = images[0]
            print(f"\n--- PVD Extraction: {sample.name} ---")
            result = extract_payload(str(sample), "pvd")
            print(f"  Quality:     {result.extraction_quality}")
            print(f"  Printable:   {result.printable_ratio:.2%}")
            print(f"  Byte Length: {result.payload_bytes_length}")
            print(f"  Obfuscation: {result.obfuscation_layer}")
            print(f"  Language:    {result.likely_language}")
            print(f"  Fallback:    {result.fallback_used}")
            if result.error:
                print(f"  Error:       {result.error}")
            snippet = result.payload_text[:200] if result.payload_text else "(empty)"
            print(f"  Payload (first 200 chars):\n    {snippet}")
        else:
            print("[SKIP] No PVD test images found")
    else:
        print(f"[SKIP] {pvd_dir} not found")

    print("\n[OK] Module 3 Payload Extractor working correctly!")
