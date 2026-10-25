"""
Traffic Sign Recognition - Simple Flask Web App
A beginner-friendly web interface for uploading a traffic sign image and getting instant predictions.
"""

import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf

app = Flask(__name__)

# Folder to temporarily save uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 43 Traffic Sign Class Names
CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed & passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

# Path to the trained model
MODEL_PATH = os.path.join('model', 'traffic_sign_model.keras')

# Load model if it exists
model = None
if os.path.exists(MODEL_PATH):
    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print("Notice: 'traffic_sign_model.keras' not found. Please run the notebook to train the model first.")

@app.route('/')
def home():
    """Renders the main upload page."""
    return render_template('index.html', model_ready=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    """Receives uploaded image, runs prediction, returns sign name and confidence."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            return jsonify({'error': 'Model file not found. Please run traffic_sign_recognition_beginner.ipynb to train the model first.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    # Save file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # 1. Read image with OpenCV
    img = cv2.imread(file_path)
    if img is None:
        return jsonify({'error': 'Failed to read image.'}), 400

    # 2. Resize to 32x32 pixels (same as during training)
    img = cv2.resize(img, (32, 32))

    # 3. Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. Normalize to 0.0 - 1.0
    img_norm = img / 255.0

    # 5. Expand dimension: (1, 32, 32, 3)
    img_batch = np.expand_dims(img_norm, axis=0)

    # 6. Predict
    probabilities = model.predict(img_batch)[0]
    predicted_class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class_id] * 100)

    # Get Top-3 predictions
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    top_3 = []
    for idx in top_3_indices:
        top_3.append({
            'class_id': int(idx),
            'name': CLASS_NAMES.get(int(idx), 'Unknown'),
            'score': round(float(probabilities[idx] * 100), 2)
        })

    return jsonify({
        'predicted_sign': CLASS_NAMES.get(predicted_class_id, 'Unknown Sign'),
        'class_id': predicted_class_id,
        'confidence': round(confidence, 2),
        'image_url': '/' + file_path.replace('\\', '/'),
        'top_3': top_3
    })

if __name__ == '__main__':
    print("Starting Traffic Sign Web App at http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
