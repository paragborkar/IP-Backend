from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from flask_cors import CORS  # Importing CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

def apply_filter(image, filter_type):
    """Apply the selected filter to the image."""
    if filter_type == 'mean':
        return cv2.blur(image, (5, 5))
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(image, 5)
    elif filter_type == 'negative':
        return cv2.bitwise_not(image)
    else:
        return image

def encode_image(image):
    """Encode the image into Base64 format."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")

@app.route("/process-image", methods=["POST"])
def process_image():
    """Process the uploaded image with the selected filter."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filter_type = request.form.get("filter", "mean")  # Default filter: mean

    # Read the image file and convert it to a numpy array
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Apply the selected filter
    processed_image = apply_filter(image, filter_type)

    # Encode the processed image into Base64 for easy transfer
    img_base64 = encode_image(processed_image)

    return jsonify({"processed_image": img_base64})

if __name__ == "__main__":
    app.run(debug=True)
