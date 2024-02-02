from preprocessing import segment

from PIL import Image
from flask import Flask, request, jsonify

import io
import cv2
import base64
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = base64.b64decode(request.json['image'])
    segmentted_img = segment(img_bytes, combine=True)
    
    # Convert the image to a base64 string
    img_str = base64.b64encode(cv2.imencode('.png', segmentted_img)[1]).decode()
    return jsonify({'segmented_image': img_str})

if __name__ == '__main__':
    app.run(debug=True)
