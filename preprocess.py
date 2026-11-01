"""
Phase 5: Preprocessing Pipeline Module
Provides consistent image transformation and dataset loading across training and inference.
"""

import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH, IMAGE_SIZE, TEST_SPLIT_RATIO, RANDOM_STATE, DATASET_DIR

PROCESSED_DATA_FILE = os.path.join(DATASET_DIR, "processed_data.npz")

def preprocess_single_image(image_path_or_array, target_size=IMAGE_SIZE):
    """
    Applies standard preprocessing to a single image:
    1. Loads from path if string, or accepts numpy array
    2. Resizes to target dimensions (32, 32)
    3. Converts BGR to RGB (if 3-channel BGR from cv2)
    4. Normalizes pixel values from [0, 255] to [0.0, 1.0]
    """
    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            raise FileNotFoundError(f"Image not found at {image_path_or_array}")
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path_or_array}")
    else:
        img = image_path_or_array.copy()

    # Resize
    img_resized = cv2.resize(img, target_size)

    # Convert BGR to RGB
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_resized

    # Normalize to [0.0, 1.0]
    img_normalized = img_rgb.astype('float32') / 255.0
    return img_normalized

def load_and_preprocess_dataset(data_path=DATA_PATH, max_classes=43):
    """
    Iterates through folders 0 to 42, reads all images, resizes to (32, 32),
    converts to RGB, normalizes, and splits into stratified train/test sets.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path '{data_path}' does not exist. Please download it first.")

    images = []
    labels = []

    print(f"Loading and preprocessing images from: {data_path}")
    for cid in range(max_classes):
        folder = os.path.join(data_path, str(cid))
        if not os.path.exists(folder):
            continue
        
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]
        for fname in files:
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is not None:
                img_32 = cv2.resize(img, IMAGE_SIZE)
                img_rgb = cv2.cvtColor(img_32, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                labels.append(cid)

    # Convert to numpy arrays
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(labels, dtype='int32')

    print(f"Loaded {len(X)} images across {max_classes} classes.")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Dataset Split: Training = {len(X_train)} samples, Testing = {len(X_test)} samples.")
    return X_train, X_test, y_train, y_test

def save_preprocessed_cache(X_train, X_test, y_train, y_test, output_file=PROCESSED_DATA_FILE):
    """Saves preprocessed arrays to compressed .npz archive for instant reuse."""
    np.savez_compressed(
        output_file,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    print(f"Cached preprocessed data saved to: {output_file}")

def load_preprocessed_cache(cache_file=PROCESSED_DATA_FILE):
    """Loads preprocessed arrays from cache if available."""
    if not os.path.exists(cache_file):
        return None
    data = np.load(cache_file)
    print(f"Loaded cached preprocessed data from: {cache_file}")
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

if __name__ == '__main__':
    # Test loading and saving cache
    X_tr, X_te, y_tr, y_te = load_and_preprocess_dataset()
    save_preprocessed_cache(X_tr, X_te, y_tr, y_te)
