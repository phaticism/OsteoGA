from preprocessing import segment, get_contours_v2, draw_points, dilate
from DCGAN import DCGAN, DropBlockNoise

from skimage.filters import gaussian
from flask import Flask, request, jsonify

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import io
import os
import cv2
import base64
import numpy as np

IMAGE_SIZE = 224

dcgan = DCGAN(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
              architecture='two-stage',
              output_activation='sigmoid',
              noise=DropBlockNoise(rate=0.1, block_size=16),
              pretrain_weights=None,
              block_type='pixel-shuffle',
              kernel_initializer='glorot_uniform',
              C=1.)


restore_model = dcgan.generator
restore_model.load_weights("./weights_gae/gan_efficientunet_full_augment-hist_equal_generator.h5")
restore_model.trainable = False


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # visualise segmentation
    img_bytes = base64.b64decode(request.json['image'])
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1), cv2.COLOR_BGR2GRAY) / 255.
    segmented_img = segment(img_bytes, combine=True)
    segmented_str = base64.b64encode(cv2.imencode('.png', segmented_img)[1]).decode()

    # visualise contours
    uc, lc = get_contours_v2(segment(img_bytes, combine=False), verbose=1)
    mask = np.zeros((640, 640)).astype('uint8')
    mask = draw_points(mask, lc, thickness=1, color=(255, 255, 255))
    mask = draw_points(mask, uc, thickness=1, color=(255, 255, 255))
    mask = cv2.resize(mask, (224, 224), cv2.INTER_NEAREST)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask / 255.
    plt.imsave('contour.png', mask, cmap='gray')
    contour_str = base64.b64encode(open('contour.png', 'rb').read()).decode()
    os.remove('contour.png')

    # visualise masked image
    mask = 1 - mask
    dilated = gaussian(dilate(mask), sigma=50, truncate=0.3)
    im = np.expand_dims(img * (1 - dilated), axis=0)
    im = tf.convert_to_tensor(im, dtype=tf.float32)
    plt.imsave('masked.png', im[0], cmap='gray')
    masked_str = base64.b64encode(open('masked.png', 'rb').read()).decode()
    os.remove('masked.png')


    # restore masked image
    restored_img = restore_model(im)
    res = tf.squeeze(tf.squeeze(restored_img, axis=-1), axis=0)
    plt.imsave('restored.png', res, cmap='gray')
    restored_str = base64.b64encode(open('restored.png', 'rb').read()).decode()
    os.remove('restored.png')

    # visualise anomaly
    plt.imsave('anomaly.png', dilated*tf.abs(img - res), cmap='turbo')
    anomaly_str = base64.b64encode(open('anomaly.png', 'rb').read()).decode()
    os.remove('anomaly.png')

    return jsonify({
        'segmented': segmented_str, 
        'contour': contour_str,
        'masked': masked_str,
        'restored': restored_str,
        'anomaly': anomaly_str,
    })

if __name__ == '__main__':
    app.run(debug=True)
