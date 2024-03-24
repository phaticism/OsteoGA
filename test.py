import requests
import json
import base64
import cv2
import os
import numpy as np


img = cv2.imread('archive/train/1/9000622R.png')
# convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode()

# Send it to the server
r = requests.post('http://localhost:8000/predict', json={'image': img_str, 'crop': 'false'})

# read the return value when code 400
if r.status_code != 200:
    print(r.json())
else:
    # Read the response
    imgs = r.json()['images']

    cropped_img = base64.b64decode(imgs['cropped']) if 'cropped' in imgs else None
    segmented_img = base64.b64decode(imgs['segmented'])
    contour_img = base64.b64decode(imgs['contour'])
    dilated_img = base64.b64decode(imgs['dilated'])
    blurred_img = base64.b64decode(imgs['blurred'])
    masked_img = base64.b64decode(imgs['masked'])
    restored_img = base64.b64decode(imgs['restored'])
    anomaly_img = base64.b64decode(imgs['anomaly'])

    if cropped_img is not None:
        cropped_img = cv2.imdecode(np.frombuffer(cropped_img, np.uint8), -1)
    segmented_img = cv2.imdecode(np.frombuffer(segmented_img, np.uint8), -1)
    contour_img = cv2.imdecode(np.frombuffer(contour_img, np.uint8), -1)
    dilated_img = cv2.imdecode(np.frombuffer(dilated_img, np.uint8), -1)
    blurred_img = cv2.imdecode(np.frombuffer(blurred_img, np.uint8), -1)
    masked_img = cv2.imdecode(np.frombuffer(masked_img, np.uint8), -1)
    restored_img = cv2.imdecode(np.frombuffer(restored_img, np.uint8), -1)
    anomaly_img = cv2.imdecode(np.frombuffer(anomaly_img, np.uint8), -1)

    if not os.path.exists('debug'):
        os.makedirs('debug')
    
    if cropped_img is not None:
        cv2.imwrite('debug/cropped.png', cropped_img)
    cv2.imwrite('debug/segmented.png', segmented_img)
    cv2.imwrite('debug/contour.png', contour_img)
    cv2.imwrite('debug/dilated.png', dilated_img)
    cv2.imwrite('debug/blurred.png', blurred_img)
    cv2.imwrite('debug/masked.png', masked_img)
    cv2.imwrite('debug/restored.png', restored_img)
    cv2.imwrite('debug/anomaly.png', anomaly_img)

    print(r.json()['probabilities'])
