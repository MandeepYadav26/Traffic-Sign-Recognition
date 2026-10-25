"""
Traffic Sign Recognition - Global Configuration File
Phase 1: Project Setup & Parameters
All hyperparameters, paths, and settings are defined here cleanly.
"""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_PATH = os.path.join(DATASET_DIR, "gtsrb")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_FILE = os.path.join(MODEL_DIR, "traffic_sign_model.keras")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# Model & Training parameters
NUM_CLASSES = 43
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Dataset download URL (Harvard CS50 clean GTSRB archive)
DATASET_URL = "https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip"
ZIP_FILE_NAME = "gtsrb.zip"

def create_project_directories():
    """Creates all required project directories if they do not exist."""
    directories = [
        DATASET_DIR,
        MODEL_DIR,
        NOTEBOOKS_DIR,
        STATIC_DIR,
        UPLOAD_DIR,
        SAMPLES_DIR,
        TEMPLATES_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Project directories verified and ready.")

if __name__ == "__main__":
    create_project_directories()
