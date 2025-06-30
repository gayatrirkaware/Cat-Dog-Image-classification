from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from datetime import datetime

# Flask app
app = Flask(__name__)

# Load trained model
model = load_model("model/MobileNet_best_tuned_model.h5")  # update path if needed

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["cat_dog_image_classification"]
collection = db["predictions"]

# Image preprocessing
def preprocess_image(file):
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_array = preprocess_image(file)
    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction >= 0.5 else "Cat"

    # Save to MongoDB
    collection.insert_one({
        "label": label,
        "probability": float(prediction),
        "timestamp": datetime.now()
    })
    
    return jsonify({"label": label, "probability": float(prediction)})

@app.route("/records")
def records():
    data = list(collection.find({}, {"_id": 0}))
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
