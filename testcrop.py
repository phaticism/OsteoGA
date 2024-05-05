import requests
import json
import base64
import cv2
import os
import numpy as np

# Load and encode the image
img = cv2.imread('test_images/my_image.png')
img_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode()

# Send it to the server
r = requests.post('http://localhost:8000/crop', json={'image': img_str})

# Check the status code
if r.status_code != 200:
    print(r.json())
else:
    # Read the response
    response = r.json()
    cropped_images = response['cropped']
    print('Number of cropped images:', response['length'])
    print('Error:', response['error'])

    # Save the cropped images
    if not os.path.exists('cropped'):
        os.makedirs('cropped')

    for i, cropped_image_str in enumerate(cropped_images):
        if cropped_image_str is not None:
            cropped_image_data = base64.b64decode(cropped_image_str)
            cropped_image = cv2.imdecode(
                np.frombuffer(cropped_image_data, np.uint8), -1)
            cv2.imwrite(f'cropped/cropped_{i}.png', cropped_image)
