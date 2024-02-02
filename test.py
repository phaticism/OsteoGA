import requests
import json
import base64
import cv2
import os
import numpy as np


img = cv2.imread('archive/val/4/9070207R.png')
img_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode()

# Send it to the server
r = requests.post('http://127.0.0.1:5000/predict', json={'image': img_str})

# Read the response
imgs = r.json()

segmented_img = base64.b64decode(imgs['segmented'])
contour_img = base64.b64decode(imgs['contour'])
masked_img = base64.b64decode(imgs['masked'])
restored_img = base64.b64decode(imgs['restored'])
anomaly_img = base64.b64decode(imgs['anomaly'])

segmented_img = cv2.imdecode(np.frombuffer(segmented_img, np.uint8), -1)
contour_img = cv2.imdecode(np.frombuffer(contour_img, np.uint8), -1)
masked_img = cv2.imdecode(np.frombuffer(masked_img, np.uint8), -1)
restored_img = cv2.imdecode(np.frombuffer(restored_img, np.uint8), -1)
anomaly_img = cv2.imdecode(np.frombuffer(anomaly_img, np.uint8), -1)

if not os.path.exists('debug'):
    os.makedirs('debug')
cv2.imwrite('debug/segmented.png', segmented_img)
cv2.imwrite('debug/contour.png', contour_img)
cv2.imwrite('debug/masked.png', masked_img)
cv2.imwrite('debug/restored.png', restored_img)
cv2.imwrite('debug/anomaly.png', anomaly_img)
