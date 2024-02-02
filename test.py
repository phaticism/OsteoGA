import requests
import json
import base64
import cv2
import numpy as np


img = cv2.imread('archive/train/2/9000296R.png')
img_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode()

# Send it to the server
r = requests.post('http://127.0.0.1:5000/predict', json={'image': img_str})


# Check the response img
response_img_str = r.json()['segmented_image']
response_img = base64.b64decode(response_img_str)
response_img = np.frombuffer(response_img, dtype=np.uint8)
response_img = cv2.imdecode(response_img, cv2.IMREAD_UNCHANGED)

# Save it
cv2.imwrite('segmented_img.png', response_img)
