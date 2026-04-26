"""
StegoDetect - Dataset Merging Script
Combines the LSB dataset (archive/) and PVD dataset (Stego-pvd-dataset/)
into a unified 3-class dataset at ./data/combined/

Classes:
    0 - clean  (clean images from both LSB and PVD datasets)
    1 - lsb    (stego, stego_b64, stego_zip from LSB dataset)
    2 - pvd    (stego images from PVD dataset)

Adapts to actual folder structure on disk:
    LSB dataset:  archive/train/train/clean, archive/train/train/stego, etc.
    PVD dataset:  Stego-pvd-dataset/train/cleanTrain, Stego-pvd-dataset/train/stegoTrain, etc.
"""

import os
import sys
import json
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime

from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Actual paths on disk
LSB_ROOT = PROJECT_ROOT / "archive"
PVD_ROOT = PROJECT_ROOT / "Stego-pvd-dataset"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "combined"

SPLITS = ["train", "test", "val"]

# Mapping: (source_path_relative_to_dataset_root, destination_class, filename_prefix)
# The LSB dataset has a doubled folder structure: train/train/, test/test/, val/val/
# The PVD dataset uses camelCase naming: cleanTrain, stegoTrain, etc.

def get_lsb_sources(split: str) -> list:
    """
    Returns list of (source_dir, dest_class, prefix) for LSB dataset.
    Adapts to the doubled-folder structure: archive/train/train/clean, etc.
    """
    sources = []
    inner = LSB_ROOT / split / split  # e.g. archive/train/train/

    # Clean images
    clean_dir = inner / "clean"
    if clean_dir.exists():
        sources.append((clean_dir, "clean", "lsb_"))

    # Stego images (raw payload) → lsb class
    stego_dir = inner / "stego"
    if stego_dir.exists():
        sources.append((stego_dir, "lsb", ""))

    # Stego base64-encoded → lsb class
    stego_b64_dir = inner / "stego_b64"
    if stego_b64_dir.exists():
        sources.append((stego_b64_dir, "lsb", "b64_"))

    # Stego zip-compressed → lsb class
    stego_zip_dir = inner / "stego_zip"
    if stego_zip_dir.exists():
        sources.append((stego_zip_dir, "lsb", "zip_"))

    return sources


def get_pvd_sources(split: str) -> list:
    """
    Returns list of (source_dir, dest_class, prefix) for PVD dataset.
    Adapts to camelCase naming: cleanTrain, stegoTrain, cleanTest, stegoTest, etc.
    """
    sources = []

    # Map split names to PVD folder suffixes
    suffix_map = {
        "train": "Train",
        "test": "Test",
        "val": "Val",
    }
    suffix = suffix_map[split]

    # Clean images
    clean_dir = PVD_ROOT / split / f"clean{suffix}"
    if clean_dir.exists():
        sources.append((clean_dir, "clean", "pvd_"))

    # Stego images → pvd class
    stego_dir = PVD_ROOT / split / f"stego{suffix}"
    if stego_dir.exists():
        sources.append((stego_dir, "pvd", ""))

    return sources


# ──────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────

def setup_logging():
    """Set up logging to file and console."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_ROOT / "merge_errors.log"

    logger = logging.getLogger("merge_datasets")
    logger.setLevel(logging.DEBUG)

    # File handler - errors only
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


# ──────────────────────────────────────────────────────────────────────
# Core merging logic
# ──────────────────────────────────────────────────────────────────────

def verify_image(path: Path) -> bool:
    """Check that an image file can be opened with Pillow."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def copy_files(
    source_dir: Path,
    dest_dir: Path,
    prefix: str,
    logger: logging.Logger,
    stats: dict,
) -> int:
    """
    Copy all image files from source_dir to dest_dir.
    
    Args:
        source_dir: Directory to copy from
        dest_dir: Directory to copy to
        prefix: Prefix to add to filenames to avoid conflicts
        logger: Logger instance
        stats: Dictionary to accumulate statistics
    
    Returns:
        Number of successfully copied files
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in source_dir.iterdir() if f.is_file()])
    copied = 0
    failed = 0

    for src_file in tqdm(files, desc=f"  {source_dir.name} → {dest_dir.name}", leave=False):
        # Build destination filename with prefix to avoid conflicts
        if prefix:
            dest_name = f"{prefix}{src_file.name}"
        else:
            dest_name = src_file.name

        dest_file = dest_dir / dest_name

        # Handle any remaining filename conflicts using hash
        if dest_file.exists():
            hash_str = hashlib.md5(str(src_file).encode()).hexdigest()[:8]
            dest_name = f"{prefix}{hash_str}_{src_file.name}"
            dest_file = dest_dir / dest_name

        try:
            shutil.copy2(src_file, dest_file)

            # Verify the copied image can be opened
            if not verify_image(dest_file):
                logger.warning(f"Corrupt image copied (failed verification): {src_file} → {dest_file}")
                stats["corrupt_files"].append(str(dest_file))
                # Keep the file anyway - it was in the original dataset

            copied += 1
        except Exception as e:
            logger.error(f"Failed to copy {src_file} → {dest_file}: {e}")
            stats["failed_copies"].append({"source": str(src_file), "error": str(e)})
            failed += 1

    return copied


def merge_datasets():
    """Main merge function."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("StegoDetect - Dataset Merging")
    logger.info("=" * 60)
    logger.info(f"LSB dataset: {LSB_ROOT}")
    logger.info(f"PVD dataset: {PVD_ROOT}")
    logger.info(f"Output:      {OUTPUT_ROOT}")
    logger.info("")

    # Verify source directories exist
    if not LSB_ROOT.exists():
        logger.error(f"LSB dataset not found at {LSB_ROOT}")
        sys.exit(1)
    if not PVD_ROOT.exists():
        logger.error(f"PVD dataset not found at {PVD_ROOT}")
        sys.exit(1)

    # Statistics tracking
    stats = {
        "start_time": datetime.now().isoformat(),
        "source_lsb": str(LSB_ROOT),
        "source_pvd": str(PVD_ROOT),
        "output": str(OUTPUT_ROOT),
        "splits": {},
        "corrupt_files": [],
        "failed_copies": [],
    }

    # Summary table data
    summary_table = {}

    for split in SPLITS:
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'─' * 40}")

        split_stats = {"clean": 0, "lsb": 0, "pvd": 0}

        # Gather all sources for this split
        all_sources = get_lsb_sources(split) + get_pvd_sources(split)

        for source_dir, dest_class, prefix in all_sources:
            if not source_dir.exists():
                logger.warning(f"  Source not found (skipping): {source_dir}")
                continue

            file_count = len([f for f in source_dir.iterdir() if f.is_file()])
            logger.info(f"  Copying {file_count} files from {source_dir.name} → {dest_class}/ (prefix='{prefix}')")

            dest_dir = OUTPUT_ROOT / split / dest_class
            copied = copy_files(source_dir, dest_dir, prefix, logger, stats)
            split_stats[dest_class] += copied

        stats["splits"][split] = split_stats
        summary_table[split] = split_stats

    # ── Create class_map.json ─────────────────────────────────────────
    class_map = {"clean": 0, "lsb": 1, "pvd": 2}
    class_map_path = OUTPUT_ROOT / "class_map.json"
    with open(class_map_path, "w") as f:
        json.dump(class_map, f, indent=4)
    logger.info(f"\nCreated {class_map_path}")

    # ── Save merge summary ────────────────────────────────────────────
    stats["end_time"] = datetime.now().isoformat()
    stats["class_map"] = class_map
    summary_path = OUTPUT_ROOT / "merge_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Saved merge summary to {summary_path}")

    # ── Print summary table ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MERGE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Split':<8} | {'Clean':>8} | {'LSB':>8} | {'PVD':>8} | {'Total':>8}")
    logger.info("-" * 52)
    grand_total = 0
    for split in SPLITS:
        s = summary_table[split]
        total = s["clean"] + s["lsb"] + s["pvd"]
        grand_total += total
        logger.info(f"{split:<8} | {s['clean']:>8} | {s['lsb']:>8} | {s['pvd']:>8} | {total:>8}")
    logger.info("-" * 52)
    logger.info(f"{'TOTAL':<8} | {sum(summary_table[s]['clean'] for s in SPLITS):>8} | "
                f"{sum(summary_table[s]['lsb'] for s in SPLITS):>8} | "
                f"{sum(summary_table[s]['pvd'] for s in SPLITS):>8} | {grand_total:>8}")

    if stats["corrupt_files"]:
        logger.info(f"\n⚠  {len(stats['corrupt_files'])} corrupt files detected (kept in dataset)")
    if stats["failed_copies"]:
        logger.info(f"\n✗  {len(stats['failed_copies'])} files failed to copy - see merge_errors.log")
    
    if not stats["corrupt_files"] and not stats["failed_copies"]:
        logger.info("\n✓  All files merged successfully with no errors!")

    logger.info(f"\nOutput directory: {OUTPUT_ROOT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    merge_datasets()
