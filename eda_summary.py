"""
Phase 4: Exploratory Data Analysis (EDA) Summary Script
Quickly computes and displays dataset statistics and class distribution from the terminal.

Usage:
    python eda_summary.py
"""

import os
import sys
import numpy as np

# Ensure root paths are available
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH
from label_utils import CLASS_NAMES

def run_eda_summary():
    print("=" * 65)
    print("🚦 PHASE 4: EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at: {DATA_PATH}")
        print("Please run 'python download_dataset.py' to acquire the dataset first.")
        print("=" * 65)
        return

    class_counts = {}
    for cid in range(43):
        folder = os.path.join(DATA_PATH, str(cid))
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]
            class_counts[cid] = len(files)
        else:
            class_counts[cid] = 0

    counts = list(class_counts.values())
    total_imgs = sum(counts)
    min_cid = min(class_counts, key=class_counts.get)
    max_cid = max(class_counts, key=class_counts.get)

    print(f"Total Traffic Sign Classes : 43")
    print(f"Total Dataset Images       : {total_imgs:,}")
    print(f"Average Images per Class   : {np.mean(counts):.1f}")
    print(f"Median Images per Class    : {np.median(counts):.1f}")
    print(f"Least Represented Class    : Class {min_cid} ({CLASS_NAMES[min_cid]}) -> {class_counts[min_cid]} images")
    print(f"Most Represented Class     : Class {max_cid} ({CLASS_NAMES[max_cid]}) -> {class_counts[max_cid]} images")
    print(f"Imbalance Ratio (Max/Min)  : {class_counts[max_cid] / max(1, class_counts[min_cid]):.1f}x")
    print("-" * 65)

    print(f"{'Class ID':<10} | {'Count':<8} | {'Traffic Sign Name'}")
    print("-" * 65)
    # Display top 5 and bottom 5 classes
    sorted_by_count = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("--- Top 5 Most Common Signs ---")
    for cid, cnt in sorted_by_count[:5]:
        print(f"Class {cid:<4} | {cnt:<8} | {CLASS_NAMES[cid]}")
        
    print("--- 5 Least Common Signs ---")
    for cid, cnt in sorted_by_count[-5:]:
        print(f"Class {cid:<4} | {cnt:<8} | {CLASS_NAMES[cid]}")

    print("=" * 65)
    print("🎉 PHASE 4 COMPLETE: Open 'notebooks/04_phase4_exploratory_data_analysis.ipynb'")
    print("   to view the distribution charts, aspect ratio plots, and full visual gallery!")
    print("=" * 65)

if __name__ == '__main__':
    run_eda_summary()
