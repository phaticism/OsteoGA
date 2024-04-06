import requests
import json
import base64
import cv2
import os
import numpy as np

img = cv2.imread('archive/train/1/9000622L.png')
# convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode()

# Send it to the server
r = requests.post('http://localhost:8000/predictall', json={'image': img_str})

# read the return value when code 400
if r.status_code != 200:
    print(r.json())
else:
    # Read the response
    response = r.json()
    objects = response['objects']
    print('Number of objects:', response['length'])
    print('Error:', response['error'])

    image_names = ['cropped', 'segmented', 'contour', 'dilated', 'blurred', 'masked', 'restored', 'anomaly']

    if not os.path.exists('debug'):
        os.makedirs('debug')

    for i, obj in enumerate(objects):
        print(f'Object {i+1} probabilities:', obj['probabilities'])
        print(f'Object {i+1} error:', obj['error'])
        imgs = obj['images']
        for image_name in image_names:
            if image_name in imgs:
                img_data = base64.b64decode(imgs[image_name])
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)
                cv2.imwrite(f'debug/{image_name}_{i+1}.png', img)