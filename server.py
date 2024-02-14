from DCGAN import DCGAN, DropBlockNoise
from Preprocessing import preprocess, segment, dilate, get_contours_v2, draw_points

from skimage.filters import gaussian
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import tensorflow as tf
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from base64 import b64encode, b64decode
import numpy as np

IMAGE_SIZE = 224
WHITE = (255, 255, 255)

model = DCGAN(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
              architecture='two-stage',
              output_activation='sigmoid',
              noise=DropBlockNoise(rate=0.1, block_size=16),
              pretrain_weights=None,
              block_type='pixel-shuffle',
              kernel_initializer='glorot_uniform',
              C=1.)
restoration_model = model.generator
restoration_model.load_weights("./weights_gae/gan_efficientunet_full_augment-hist_equal_generator.h5")
restoration_model.trainable = False


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

def simulate():
    return np.random.dirichlet(np.ones(3), size=1).squeeze()


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    print("Request received", request.json['image'][:10])
    img_bytes = b64decode(request.json['image']) # get image bytes
    original = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    if len(original.shape) == 3 and original.shape[2] == 3:
        # Convert the image to grayscale
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    original = preprocess(cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST))
    # do segmentation
    segmented_img = cv2.resize(segment(img_bytes, combine=True), (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST)
    segmented_str = b64encode(cv2.imencode('.png', segmented_img)[1]).decode()

    # get contours
    uc, lc = get_contours_v2(segment(img_bytes, combine=False), verbose=0)
    mask = draw_points(np.zeros((640, 640)).astype('uint8'), lc, thickness=1, color=WHITE)
    mask = draw_points(mask, uc, thickness=1, color=WHITE)
    mask = cv2.cvtColor(cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY) / 255.
    ok, buffer = cv2.imencode(".png", (mask * 255).astype('uint8'))
    if ok:
        contour_str = b64encode(buffer).decode()

    # create masked image
    mask = 1 - mask
    dilated = gaussian(dilate(mask), sigma=50, truncate=0.3)
    masked_img = original * (1 - dilated)
    ok, buffer = cv2.imencode(".png", (masked_img * 255).astype('uint8'))
    if ok:
        masked_str = b64encode(buffer).decode()

    # restore masked image using generator
    input = tf.convert_to_tensor(np.expand_dims(masked_img, axis=0), dtype=tf.float32)
    restored_img = restoration_model(input)
    restored_img = tf.squeeze(tf.squeeze(restored_img, axis=-1), axis=0)
    ok, buffer = cv2.imencode(".png", (restored_img * 255).numpy().astype('uint8'))
    if ok:
        restored_str = b64encode(buffer).decode()

    # evaluate anomaly map
    anomaly_map = dilated * tf.abs(original - restored_img)
    plt.imsave('anomaly.png', anomaly_map, cmap='turbo')
    anomaly_str = b64encode(open('anomaly.png', 'rb').read()).decode()
    os.remove('anomaly.png')

    return jsonify({
        'images': {
            'segmented': segmented_str, 
            'contour': contour_str,
            'masked': masked_str,
            'restored': restored_str,
            'anomaly': anomaly_str,
        },
        'probabilities': simulate().tolist(),
    })

if __name__ == '__main__':
    app.run(debug=True)
