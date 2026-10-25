"""
Phase 9: Standalone Image Prediction Script
Loads the trained model and predicts the traffic sign category for any input image.

Usage:
    python predict.py path/to/sign_image.png
"""

import os
import sys
import json
import cv2
import numpy as np

MODEL_PATH = os.path.join("model", "traffic_sign_model.keras")
LABELS_PATH = "class_labels.json"

def load_class_labels():
    """Load class label dictionary from JSON file."""
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
    return {}

def predict_image(image_path, top_k=3):
    """Loads image, preprocesses it, runs model prediction, and displays top results."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at '{MODEL_PATH}'.")
        print("Please run 'traffic_sign_recognition_beginner.ipynb' to train the model first.")
        return

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'.")
        return

    # Import TensorFlow only when running prediction to keep startup fast
    import tensorflow as tf

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_class_labels()

    # Preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'.")
        return

    # 1. Resize to 32x32
    img_resized = cv2.resize(img, (32, 32))
    # 2. Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 3. Normalize to [0.0, 1.0]
    img_norm = img_rgb / 255.0
    # 4. Expand dimension to batch of 1: (1, 32, 32, 3)
    input_batch = np.expand_dims(img_norm, axis=0)

    # 5. Predict
    probabilities = model.predict(input_batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    best_class = top_indices[0]
    best_conf = probabilities[best_class] * 100
    best_name = labels.get(best_class, f"Class {best_class}")

    print("\n" + "=" * 45)
    print("🚦 TRAFFIC SIGN PREDICTION RESULT")
    print("=" * 45)
    print(f"Input Image : {image_path}")
    print(f"Predicted   : {best_name}")
    print(f"Class ID    : {best_class}")
    print(f"Confidence  : {best_conf:.2f}%")
    print("-" * 45)
    print(f"Top {top_k} Predictions:")
    for rank, idx in enumerate(top_indices, 1):
        name = labels.get(idx, f"Class {idx}")
        score = probabilities[idx] * 100
        print(f"  {rank}. [{score:5.2f}%] (ID {idx:2d}) {name}")
    print("=" * 45 + "\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Default test look-up
        sample_dir = os.path.join("dataset", "gtsrb", "14")
        if os.path.exists(sample_dir):
            sample_files = os.listdir(sample_dir)
            if sample_files:
                test_image = os.path.join(sample_dir, sample_files[0])
            else:
                test_image = "sample.png"
        else:
            test_image = "sample.png"
            print("Notice: No image path provided. Example usage: python predict.py sample.png")

    predict_image(test_image)
