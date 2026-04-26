"""
StegoDetect - Dataset Verification Script
Verifies dataset integrity by checking:
    - All images are exactly 512x512 pixels
    - All images are PNG format
    - All images can be opened without errors
    - Reports any issues found with pass/fail summary

Usage:
    python data_prep/verify_dataset.py               # Verify raw datasets
    python data_prep/verify_dataset.py --dir ./data/combined/  # Verify merged dataset
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_directory(root_dir: Path, expected_size=(512, 512)) -> dict:
    """
    Verify all images in a directory tree.
    
    Returns:
        dict with verification results
    """
    results = {
        "total_files": 0,
        "valid_images": 0,
        "wrong_size": [],
        "wrong_format": [],
        "corrupted": [],
        "non_image": [],
        "folder_counts": defaultdict(int),
    }

    # Collect all files
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            filepath = Path(dirpath) / fname
            all_files.append(filepath)

    results["total_files"] = len(all_files)

    if not all_files:
        print(f"  ⚠  No files found in {root_dir}")
        return results

    for filepath in tqdm(all_files, desc=f"Verifying {root_dir.name}", leave=True):
        # Track folder counts
        rel_path = filepath.relative_to(root_dir)
        folder_key = str(rel_path.parent)
        results["folder_counts"][folder_key] += 1

        # Check file extension
        if filepath.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            results["non_image"].append(str(filepath))
            continue

        if filepath.suffix.lower() != ".png":
            results["wrong_format"].append(str(filepath))

        # Try to open and verify
        try:
            with Image.open(filepath) as img:
                # Check size
                if img.size != expected_size:
                    results["wrong_size"].append({
                        "file": str(filepath),
                        "actual_size": img.size,
                    })
                else:
                    results["valid_images"] += 1
        except Exception as e:
            results["corrupted"].append({
                "file": str(filepath),
                "error": str(e),
            })

    return results


def print_report(results: dict, label: str):
    """Print a formatted verification report."""
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION REPORT: {label}")
    print(f"{'=' * 60}")

    print(f"\nTotal files scanned: {results['total_files']}")
    print(f"Valid images (512x512 PNG): {results['valid_images']}")

    # Folder breakdown
    if results["folder_counts"]:
        print(f"\nFolder breakdown:")
        for folder, count in sorted(results["folder_counts"].items()):
            print(f"  {folder}: {count} files")

    # Issues
    issues_found = False

    if results["wrong_size"]:
        issues_found = True
        print(f"\n✗ Wrong size: {len(results['wrong_size'])} files")
        for item in results["wrong_size"][:5]:  # Show first 5
            print(f"    {item['file']} → {item['actual_size']}")
        if len(results["wrong_size"]) > 5:
            print(f"    ... and {len(results['wrong_size']) - 5} more")

    if results["wrong_format"]:
        issues_found = True
        print(f"\n✗ Non-PNG format: {len(results['wrong_format'])} files")
        for f in results["wrong_format"][:5]:
            print(f"    {f}")
        if len(results["wrong_format"]) > 5:
            print(f"    ... and {len(results['wrong_format']) - 5} more")

    if results["corrupted"]:
        issues_found = True
        print(f"\n✗ Corrupted: {len(results['corrupted'])} files")
        for item in results["corrupted"][:5]:
            print(f"    {item['file']}: {item['error']}")
        if len(results["corrupted"]) > 5:
            print(f"    ... and {len(results['corrupted']) - 5} more")

    if results["non_image"]:
        print(f"\n⚠  Non-image files: {len(results['non_image'])} files")
        for f in results["non_image"][:5]:
            print(f"    {f}")

    # Final verdict
    print(f"\n{'─' * 60}")
    if not issues_found:
        print("✓ PASS - All images verified successfully!")
    else:
        total_issues = len(results["wrong_size"]) + len(results["wrong_format"]) + len(results["corrupted"])
        print(f"✗ FAIL - {total_issues} issue(s) found")
    print(f"{'=' * 60}\n")

    return not issues_found


def main():
    parser = argparse.ArgumentParser(description="Verify StegoDetect dataset integrity")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory to verify. If not provided, verifies raw datasets.",
    )
    args = parser.parse_args()

    all_passed = True

    if args.dir:
        # Verify a specific directory
        target = Path(args.dir)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        if not target.exists():
            print(f"✗ Directory not found: {target}")
            sys.exit(1)

        results = verify_directory(target)
        passed = print_report(results, str(target))
        all_passed = all_passed and passed

    else:
        # Verify raw datasets
        print("Verifying raw datasets...\n")

        # LSB dataset (archive/)
        lsb_root = PROJECT_ROOT / "archive"
        if lsb_root.exists():
            results = verify_directory(lsb_root)
            passed = print_report(results, "LSB Dataset (archive/)")
            all_passed = all_passed and passed
        else:
            print(f"⚠  LSB dataset not found at {lsb_root}")
            all_passed = False

        # PVD dataset
        pvd_root = PROJECT_ROOT / "Stego-pvd-dataset"
        if pvd_root.exists():
            results = verify_directory(pvd_root)
            passed = print_report(results, "PVD Dataset (Stego-pvd-dataset/)")
            all_passed = all_passed and passed
        else:
            print(f"⚠  PVD dataset not found at {pvd_root}")
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
