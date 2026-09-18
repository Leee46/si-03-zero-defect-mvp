from flask import Flask, request, jsonify
from models.defect_model import load_model
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# Load model
defect_model = load_model()

@app.route('/detect', methods=['POST'])
def detect_defect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = defect_model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]

    return jsonify({
        "class": class_id,
        "confidence": float(confidence),
        "status": "success"
    })
