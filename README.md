# 🐱🐶 Cat vs. Dog Image Classifier Web App

This is a Flask-based web application that classifies uploaded images as either a **Cat** or a **Dog** using a fine-tuned **MobileNet** deep learning model. Each prediction is stored in a **MongoDB** collection along with its confidence score and timestamp.

---

## 🚀 Features

- Upload and classify images as **Cat** or **Dog**
- Real-time prediction using a pretrained Keras model
- MongoDB integration for storing prediction history
- RESTful API with JSON responses
- Simple HTML frontend for manual uploads

---

## 🧠 Model Details

- Model Architecture: **MobileNet**
- Input: 224x224 RGB images
- Output: Binary classification (`Dog` if prediction ≥ 0.5, else `Cat`)
- Format: `MobileNet_best_tuned_model.h5`

---

## 🗂️ Project Structure

├── app.py # Main Flask application
├── model/
│ └── MobileNet_best_tuned_model.h5 # Trained binary classifier
├── templates/
│ └── index.html # HTML form (optional frontend)
├── static/ # Optional for CSS/JS files
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cat-dog-classifier.git
cd cat-dog-classifier
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Sample requirements.txt:
Flask
numpy
opencv-python
tensorflow
pymongo


### 🐾 Running the App
1. Start MongoDB
Ensure MongoDB is installed and running:
```bash
mongod --dbpath "C:/data/db"  # Adjust path if needed
```

2. Start the Flask App
```bash
python app.py
```

Visit http://127.0.0.1:5000 in your browser.