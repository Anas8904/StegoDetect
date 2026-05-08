"""
StegoDetect - NLP Corpus Builder

Extracts payloads from stego images to build training data for Module 4 NLP.
Runs Module 3 extraction on the stego dataset and saves results as CSV.

Output: data/nlp_corpus/train.csv, data/nlp_corpus/val.csv, data/nlp_corpus/test.csv
Columns: text, label, obfuscation_type, source_image, technique

Usage:
    python data_prep/build_nlp_corpus.py
"""

import sys
import csv
import os
from pathlib import Path
from collections import Counter

from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from modules.module3_extractor import extract_payload


# Label mapping based on payload content heuristics
def assign_label(payload_text, filename):
    """
    Assign NLP label based on payload content analysis.
    Returns label index (0-4) or -1 if unclassifiable.
    """
    import re

    if not payload_text or len(payload_text.strip()) < 5:
        return -1

    text = payload_text.strip()
    text_lower = text.lower()

    # Ethereum address
    if re.search(r'0x[0-9a-fA-F]{40}', text):
        return 4  # Ethereum_Address

    # PowerShell
    ps_kw = ["invoke-", "$env:", "get-", "set-", "new-object",
             "-encodedcommand", "powershell", "iex(", "downloadstring",
             "system.net", "start-process", "[system.", "add-type"]
    if sum(1 for kw in ps_kw if kw in text_lower) >= 2:
        return 2  # PowerShell

    # HTML with JavaScript
    html_kw = ["<script", "<html", "<iframe", "onclick=", "<div",
               "<body", "document.write"]
    if sum(1 for kw in html_kw if kw in text_lower) >= 1:
        return 1  # JavaScript_HTML

    # JavaScript
    js_kw = ["function", "eval(", "document.", "=>", "var ", "const ",
             "let ", "window.", "settimeout", "require("]
    if sum(1 for kw in js_kw if kw in text_lower) >= 2:
        return 0  # JavaScript

    # URL/IP
    if re.search(r'https?://', text) or re.search(r'(?:\d{1,3}\.){3}\d{1,3}', text):
        return 3  # URL_IP

    # If has enough printable text, try to assign based on any single keyword
    if len(text) > 20:
        if any(kw in text_lower for kw in ps_kw):
            return 2
        if any(kw in text_lower for kw in js_kw):
            return 0
        if any(kw in text_lower for kw in html_kw):
            return 1

    return -1  # unclassifiable


def build_corpus():
    """Build NLP training corpus from stego images."""
    print("StegoDetect - NLP Corpus Builder")
    print("=" * 50)

    corpus_dir = config.NLP_CORPUS_DIR
    corpus_dir.mkdir(parents=True, exist_ok=True)

    label_names = {0: "JavaScript", 1: "JavaScript_HTML", 2: "PowerShell",
                   3: "URL_IP", 4: "Ethereum_Address"}

    for split in ["train", "val", "test"]:
        print(f"\n--- Processing {split} split ---")

        output_path = corpus_dir / f"{split}.csv"
        rows = []

        for technique in ["lsb", "pvd"]:
            stego_dir = config.COMBINED_DIR / split / technique
            if not stego_dir.exists():
                print(f"  [SKIP] {stego_dir} not found")
                continue

            images = sorted([
                f for f in stego_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ])
            print(f"  {technique.upper()}: {len(images)} images")

            for img_path in tqdm(images, desc=f"  {technique}", leave=False):
                try:
                    result = extract_payload(str(img_path), technique)

                    if result.extraction_quality == "failed":
                        continue
                    if not result.payload_text or len(result.payload_text.strip()) < 10:
                        continue

                    # Determine obfuscation type from filename
                    obf = "none"
                    if img_path.name.startswith("b64_"):
                        obf = "base64"
                    elif img_path.name.startswith("zip_"):
                        obf = "zip"

                    label = assign_label(result.payload_text, img_path.name)
                    if label == -1:
                        continue

                    # Truncate for safety
                    text = result.payload_text[:10000]

                    rows.append({
                        "text": text,
                        "label": label,
                        "label_name": label_names[label],
                        "obfuscation_type": obf,
                        "source_image": img_path.name,
                        "technique": technique,
                        "quality": result.extraction_quality,
                    })

                except Exception:
                    continue

        # Write CSV
        if rows:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            # Stats
            label_counts = Counter(r["label_name"] for r in rows)
            print(f"\n  Saved {len(rows)} samples to {output_path}")
            for name, count in sorted(label_counts.items()):
                print(f"    {name}: {count}")
        else:
            print(f"\n  [WARN] No valid samples extracted for {split}")

    print("\n[OK] NLP corpus building complete!")
    print(f"     Output directory: {corpus_dir}")


if __name__ == "__main__":
    build_corpus()
