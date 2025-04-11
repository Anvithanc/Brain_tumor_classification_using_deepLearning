# Starting Flask App
print("Starting Flask App...")

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained deep learning model
MODEL_PATH = "C:\\Users\\RAJATH\\OneDrive\\Desktop\\final ml\\trained_model_resnet.h5"
model = load_model(MODEL_PATH)

# Define class labels for brain tumor types
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

# Ensure the 'static' folder exists for uploaded files
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to display the upload form
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route to handle image upload and make predictions
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
    file.save(file_path)

    # Preprocess the image
    img = image.load_img(file_path, target_size=(150, 150))  # Adjust image size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    # Return results to the frontend
    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence,
        "image_path": file_path
    })

# Run the Flask app
if __name__ == "__main__":
    print("Flask app is starting...")
    app.run(debug=True, host="127.0.0.1", port=5000)
