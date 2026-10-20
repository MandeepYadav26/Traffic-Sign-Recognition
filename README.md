# 🚦 Traffic Sign Recognition System (Beginner Friendly)

A simple Deep Learning project for B.Tech students using **Convolutional Neural Networks (CNN)** and **TensorFlow/Keras** to classify 43 types of traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB).

---

## 📁 Project Structure

```
Traffic sign recognition/
│
├── traffic_sign_recognition_beginner.ipynb   <-- 🌟 Run this Jupyter Notebook first!
├── app.py                                    <-- Simple Flask web app
├── templates/
│   └── index.html                            <-- Webpage for uploading and testing images
├── requirements.txt                          <-- Required Python libraries
└── README.md                                 <-- Instructions (You are here!)
```

---

## ⚡ Quick Start: 3 Simple Steps

### Step 1: Install Dependencies
Open your terminal / command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Open and Run the Jupyter Notebook
Open [`traffic_sign_recognition_beginner.ipynb`](traffic_sign_recognition_beginner.ipynb) in **VS Code** or upload it to **Google Colab**.

Click **"Run All"** (or run cells one by one):
1. **Downloads the dataset automatically** (no manual downloading required!).
2. Loads and resizes traffic sign images to $32 \times 32$ pixels.
3. Shows sample images of traffic signs.
4. Trains a simple, beginner-friendly CNN model in 15 epochs.
5. Displays accuracy & loss graphs.
6. Saves the trained model to `model/traffic_sign_model.keras`.

### Step 3: Run the Web App
Once the notebook has saved the model, run the Flask web application:
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

Upload any traffic sign image and click **"Predict Traffic Sign"** to see:
- Predicted sign name (e.g. *Speed Limit 50 km/h*)
- Class ID
- Confidence percentage
- Top-3 candidate predictions with progress bars

---

## 🌐 Dataset Reference
This project uses the clean Harvard CS50 GTSRB archive containing 43 classes (`0` to `42`):
- Direct Download Link: `https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip`
- Handled automatically inside the notebook.
