"""
Phase 1: Environment & Project Verification Script
Run this script to check if all libraries and folders are ready for Traffic Sign Recognition.

Usage:
    python verify_phase1.py
"""

import sys
import os
import platform

def verify_environment():
    print("=" * 60)
    print("🚦 PHASE 1: ENVIRONMENT & DIRECTORY VERIFICATION")
    print("=" * 60)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Platform          : {platform.platform()}")
    print(f"Working Directory : {os.getcwd()}")
    print("-" * 60)

    # 1. Check Libraries
    packages = {
        "NumPy": "numpy",
        "OpenCV": "cv2",
        "Matplotlib": "matplotlib",
        "Scikit-Learn": "sklearn",
        "TensorFlow": "tensorflow",
        "Flask": "flask"
    }

    missing = []
    print("Checking Dependencies:")
    for display_name, pkg in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [OK] {display_name:<15} : {version}")
        except ImportError:
            print(f"  [MISSING] {display_name}")
            missing.append(pkg)

    print("-" * 60)

    # 2. Check GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Hardware Acceleration : GPU Enabled ({len(gpus)} device(s))")
        else:
            print("Hardware Acceleration : Running on CPU (Fully sufficient for 32x32 CNN)")
    except Exception:
        pass

    print("-" * 60)

    # 3. Create & Verify Directories
    from config import create_project_directories, DATASET_DIR, MODEL_DIR, NOTEBOOKS_DIR, UPLOAD_DIR
    create_project_directories()

    print("-" * 60)
    if missing:
        print(f"⚠️ Action Required: Install missing packages using:")
        print(f"   pip install {' '.join(missing)}")
    else:
        print("🎉 PHASE 1 COMPLETE: Environment and project directories are ready for Phase 2!")
    print("=" * 60)

if __name__ == '__main__':
    verify_environment()
