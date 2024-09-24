import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import cv2
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('eye_disease_model.h5')

# Class labels for mapping output to disease names
class_labels = {
    0: "Bulging_Eyes",
    1: "Cataracts",
    2: "Crossed_Eyes",
    3: "Glaucoma",
    4: "Uveitis",
    5: "No disease"
}

img_rows = 224
img_cols = 224

def preprocess_image(image):
    """
    Preprocess the input image to the size the model expects (e.g., 224x224).
    """
    # Resize the image
    img = cv2.resize(image, (img_rows, img_cols))  
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension for prediction
    return img

def predict_eye_disease(image):
    """
    Predicts the eye disease based on the input image.
    """
    # Preprocess the image
    img = preprocess_image(image)
    
    # Get model predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])  # Get the class with the highest score
    
    # Map the predicted class to the eye disease or "No disease"
    result = class_labels[predicted_class]
    
    if result == "No disease":
        return "No eye disease detected"
    else:
        return f"Eye disease detected: {result}"

# Route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Convert the file to a JPEG image
    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')  # Convert to RGB (if not already in that format)
        img = np.array(img)  # Convert to numpy array for processing

        # Predict the eye disease
        result = predict_eye_disease(img)

        # Return the result as JSON
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
