"""
Phase 2: Dataset Acquisition Script
Downloads and extracts the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
Source: Harvard CS50 GTSRB Clean Archive (~200MB, 43 classes: folders 0 to 42).

Usage:
    python download_dataset.py
"""

import os
import sys
import urllib.request
import zipfile

from config import DATASET_DIR, DATA_PATH, DATASET_URL, ZIP_FILE_NAME

def download_progress_hook(block_num, block_size, total_size):
    """Simple download progress bar display."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_tot = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading GTSRB: {percent:5.1f}% ({mb_down:6.1f} / {mb_tot:6.1f} MB)")
        sys.stdout.flush()

def main():
    print("=" * 60)
    print(" PHASE 2: GTSRB DATASET ACQUISITION & EXTRACTION")
    print("=" * 60)

    # 1. Check if dataset already exists
    if os.path.exists(DATA_PATH):
        classes = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
        if len(classes) >= 43:
            print(f" Dataset already downloaded and verified at:\n   {DATA_PATH}")
            print(f"   Found {len(classes)} class folders.")
            print("=" * 60)
            return

    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(DATASET_DIR, ZIP_FILE_NAME)

    # 2. Download
    if not os.path.exists(zip_path):
        print(f"Downloading from: {DATASET_URL}")
        print("This file is ~200MB. Please wait a few moments...\n")
        urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=download_progress_hook)
        print("\n\n Download complete!")
    else:
        print(f"Found existing zip file: {zip_path}")

    # 3. Extract
    print("Extracting files to dataset folder...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print("Extraction finished!")

    # 4. Clean up zip
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("Removed temporary archive to save space.")

    # 5. Verification
    if os.path.exists(DATA_PATH):
        folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
        print(f"\n PHASE 2 COMPLETE: Successfully extracted {len(folders)} class folders!")
        print(f"Dataset location: {os.path.abspath(DATA_PATH)}")
    else:
        print("\n Warning: Could not locate extracted 'gtsrb' directory.")
    print("=" * 60)

if __name__ == '__main__':
    main()
