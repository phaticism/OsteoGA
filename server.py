import logging
import numpy as np
from base64 import b64encode, b64decode
import cv2
import torch
from DCGAN import DCGAN, DropBlockNoise
from Preprocessing import preprocess, segment, dilate, get_contours_v2, draw_points
from NewClassifier import Classifier
from skimage.filters import gaussian
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import matplotlib.pyplot as plt
import absl.logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

absl.logging.set_verbosity(absl.logging.ERROR)


logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

IMAGE_SIZE = 224
WHITE = (255, 255, 255)


def get_extraction_model():
    model = torch.hub.load(
        './yolov5', 'custom', path='./yolov5/runs/train/exp/weights/best.pt', source='local')
    return model


extraction_model = get_extraction_model()

model = DCGAN(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
              architecture='two-stage',
              output_activation='sigmoid',
              noise=DropBlockNoise(rate=0.1, block_size=16),
              pretrain_weights=None,
              block_type='pixel-shuffle',
              kernel_initializer='glorot_uniform',
              C=1.)
restoration_model = model.generator
restoration_model.load_weights(
    "./weights_gae/gan_efficientunet_full_augment-hist_equal_generator.h5")
restoration_model.trainable = False


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


classifier = Classifier('weights_5cls/svc.pkl',
                        'weights_5cls/efficientnetv2s-subset.weights.h5')


def make_json_response_with_status(images_dict, probabilities, error, status_code, results=None, explanation_image=None):
    response = jsonify({
        'images': images_dict,
        'probabilities': probabilities if probabilities is not None else [],
        'results': results if results is not None else {},
        'error': error,
    })
    return make_response(response, status_code)


@app.route('/')
@cross_origin()
def index():
    return 'Server is running!'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    logger.info(f'Request received: {request.json["image"][-10:]}')
    images_dict = dict()

    if 'clinical' not in request.json\
            or 'age' not in request.json['clinical']\
            or 'height' not in request.json['clinical']\
            or 'weight' not in request.json['clinical']\
            or 'max_weight' not in request.json['clinical']:
        return make_json_response_with_status(images_dict, [], 'missing_clinical_data', 500)
    age = request.json['clinical']['age']
    bmi = request.json['clinical']['weight'] / \
        (request.json['clinical']['height'] / 100)**2
    max_weight = request.json['clinical']['max_weight']

    if 'crop' not in request.json:
        request.json['crop'] = 'false'
    else:
        request.json['crop'] = request.json['crop'].lower()

    try:
        img_bytes = b64decode(request.json['image'])  # get image bytes
        original = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)

        if request.json['crop'] == 'true':
            results = extraction_model(original)
            bboxes = results.pandas().xyxy[0].values.tolist()

            if len(bboxes) == 0:
                logger.info('No object detected!')
                return make_json_response_with_status(images_dict, [], 'no_object_detected', 500)
            coordinates = []
            for object in bboxes:
                coordinates.append([int(object[0]), int(
                    object[1]), int(object[2]), int(object[3])])

            x1, y1, x2, y2 = coordinates[0][0:4]
            cropped_image = original[y1:y2, x1:x2, ...]
            cropped_image = cv2.resize(cropped_image, (224, 224))

            cropped_str = b64encode(cv2.imencode(
                '.png', cropped_image)[1]).decode()
            images_dict['cropped'] = cropped_str

            original = cropped_image
            img_bytes = b64decode(cropped_str)

        # convert to grayscale if necessary
        if len(original.shape) == 3 and original.shape[2] == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        original_classification = original.copy()
        original = preprocess(cv2.resize(
            original, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST))
    except Exception as e:
        logger.error(e)
        return make_json_response_with_status(images_dict, [], 'invalid_input_format', 500)

    # perform segmentation
    try:
        segmented_img = cv2.resize(
            segment(img_bytes, combine=True), (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST)
        segmented_str = b64encode(cv2.imencode(
            '.png', segmented_img)[1]).decode()
        images_dict['segmented'] = segmented_str
    except Exception as e:
        logger.error(e)
        return make_json_response_with_status(images_dict, [], 'segmentation_failed', 500)

    # get contours
    try:
        uc, lc = get_contours_v2(segment(img_bytes, combine=False), verbose=0)
        mask = draw_points(np.zeros((640, 640)).astype(
            'uint8'), lc, thickness=1, color=WHITE)
        mask = draw_points(mask, uc, thickness=1, color=WHITE)
        mask = cv2.cvtColor(cv2.resize(
            mask, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY) / 255.
        ok, buffer = cv2.imencode(".png", (mask * 255).astype('uint8'))
        if ok:
            contour_str = b64encode(buffer).decode()
            images_dict['contour'] = contour_str
    except Exception as e:
        logger.error(e)
        return make_json_response_with_status(images_dict, [], 'contour_extraction_failed', 500)

    # create masked image
    try:
        mask = 1 - mask
        dilated = dilate(mask)
        ok, buffer = cv2.imencode(".png", (dilated * 255).astype('uint8'))
        if ok:
            dilated_str = b64encode(buffer).decode()
            images_dict['dilated'] = dilated_str

        blurred = gaussian(dilated, sigma=50, truncate=0.3)
        ok, buffer = cv2.imencode(".png", (blurred * 255).astype('uint8'))
        if ok:
            blurred_str = b64encode(buffer).decode()
            images_dict['blurred'] = blurred_str

        masked_img = original * (1 - blurred)
        ok, buffer = cv2.imencode(".png", (masked_img * 255).astype('uint8'))
        if ok:
            masked_str = b64encode(buffer).decode()
            images_dict['masked'] = masked_str
    except Exception as e:
        logger.error(e)
        return make_json_response_with_status(images_dict, [], 'masking_failed', 500)

    # restore masked image using generator
    try:
        input = tf.convert_to_tensor(np.expand_dims(
            masked_img, axis=0), dtype=tf.float32)
        restored_img = restoration_model(input)
        restored_img = tf.squeeze(tf.squeeze(restored_img, axis=-1), axis=0)
        ok, buffer = cv2.imencode(
            ".png", (restored_img * 255).numpy().astype('uint8'))
        if ok:
            restored_str = b64encode(buffer).decode()
            images_dict['restored'] = restored_str
    except Exception as e:
        logger.error(e)
        return make_json_response_with_status(images_dict, [], 'restoration_failed', 500)

    # evaluate anomaly map
    try:
        anomaly_map = blurred * tf.abs(original - restored_img)
        plt.imsave('anomaly.png', anomaly_map, cmap='turbo', vmax=0.7)
        anomaly_str = b64encode(open('anomaly.png', 'rb').read()).decode()
        images_dict['anomaly'] = anomaly_str
        os.remove('anomaly.png')
    except Exception as e:
        logger.error(e)
        return make_json_response_with_status(images_dict, [], 'anomaly_map_failed', 500)

    # classify the restored image
    try:
        results, explain_image_html = classifier.predict_proba(
            original_classification, age, max_weight, bmi)
    except Exception as e:
        logger.error(e)
        return make_json_response_with_status(images_dict, [], 'classification_failed', 500)

    logger.info('Request processed successfully!')
    probabilities = results['kl_grade']
    images_dict['explanation'] = explain_image_html
    return make_json_response_with_status(images_dict, probabilities, None, 200, results, explain_image_html)


def create_object():
    return {
        'images': {
            'cropped': '',
            'segmented': '',
            'contour': '',
            'dilated': '',
            'blurred': '',
            'masked': '',
            'restored': '',
            'anomaly': '',
            'explanation': '',
        },
        'probabilities': [],
        'error': None,
        'results': {},
    }


@app.route('/predictall', methods=['POST'])
@cross_origin()
def predictall():
    logger.info(f'Request received: {request.json["image"][-10:]}')
    if 'image' not in request.json:
        return make_response(jsonify({
            'objects': [
                create_object(),
                create_object(),
            ],
            'length': 0,
            'error': 'invalid_input_format'
        }), 500)

    if 'clinical' not in request.json\
            or 'age' not in request.json['clinical']\
            or 'height' not in request.json['clinical']\
            or 'weight' not in request.json['clinical']\
            or 'max_weight' not in request.json['clinical']:
        return make_response(jsonify({
            'objects': [
                create_object(),
                create_object(),
            ],
            'length': 0,
            'error': 'missing_clinical_data'
        }), 500)
    age = request.json['clinical']['age']
    bmi = request.json['clinical']['weight'] / \
        (request.json['clinical']['height'] / 100)**2
    max_weight = request.json['clinical']['max_weight']

    try:
        img_bytes = b64decode(request.json['image'])  # get image bytes
        original = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        original_copy = original.copy()

        results = extraction_model(original)
        bboxes = results.pandas().xyxy[0].values.tolist()

        if len(bboxes) == 0:
            logger.info('No object detected!')
            return make_response(jsonify({
                'objects': [
                    create_object(),
                    create_object(),
                ],
                'length': 0,
                'error': 'no_object_detected'
            }), 500)
        coordinates = []
        for object in bboxes:
            coordinates.append([int(object[0]), int(
                object[1]), int(object[2]), int(object[3])])
        print(len(coordinates))
    except Exception as e:
        logger.error(e)
        return make_response(jsonify({
            'objects': [
                create_object(),
                create_object(),
            ],
            'length': 0,
            'error': 'invalid_input_format'
        }), 500)

    objects = []
    for i in range(len(coordinates)):
        object = create_object()
        original = original_copy.copy()
        try:
            x1, y1, x2, y2 = coordinates[i][0:4]
            cropped_image = original[y1:y2, x1:x2, ...]
            cropped_image = cv2.resize(cropped_image, (224, 224))
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            cropped_str = b64encode(cv2.imencode(
                '.png', cropped_image)[1]).decode()
            object['images']['cropped'] = cropped_str

            original = cropped_image
            img_bytes = b64decode(cropped_str)

            original_classification = original.copy()
            original = preprocess(cv2.resize(
                original, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST))
        except Exception as e:
            logger.error(e)
            object['probabilities'] = []
            object['error'] = 'cropping_failed'
            objects.append(object)
            continue

        try:
            segmented_img = cv2.resize(
                segment(img_bytes, combine=True), (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST)
            segmented_str = b64encode(cv2.imencode(
                '.png', segmented_img)[1]).decode()
            object['images']['segmented'] = segmented_str
        except Exception as e:
            logger.error(e)
            object['probabilities'] = []
            object['error'] = 'segmentation_failed'
            objects.append(object)
            continue

        try:
            uc, lc = get_contours_v2(
                segment(img_bytes, combine=False), verbose=0)
            mask = draw_points(np.zeros((640, 640)).astype(
                'uint8'), lc, thickness=1, color=WHITE)
            mask = draw_points(mask, uc, thickness=1, color=WHITE)
            mask = cv2.cvtColor(cv2.resize(
                mask, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY) / 255.
            ok, buffer = cv2.imencode(".png", (mask * 255).astype('uint8'))
            if ok:
                contour_str = b64encode(buffer).decode()
                object['images']['contour'] = contour_str
        except Exception as e:
            logger.error(e)
            object['probabilities'] = []
            object['error'] = 'contour_extraction_failed'
            objects.append(object)
            continue

        try:
            mask = 1 - mask
            dilated = dilate(mask)
            ok, buffer = cv2.imencode(".png", (dilated * 255).astype('uint8'))
            if ok:
                dilated_str = b64encode(buffer).decode()
                object['images']['dilated'] = dilated_str

            blurred = gaussian(dilated, sigma=50, truncate=0.3)
            ok, buffer = cv2.imencode(".png", (blurred * 255).astype('uint8'))
            if ok:
                blurred_str = b64encode(buffer).decode()
                object['images']['blurred'] = blurred_str

            masked_img = original * (1 - blurred)
            ok, buffer = cv2.imencode(
                ".png", (masked_img * 255).astype('uint8'))
            if ok:
                masked_str = b64encode(buffer).decode()
                object['images']['masked'] = masked_str
        except Exception as e:
            logger.error(e)
            object['probabilities'] = []
            object['error'] = 'masking_failed'
            objects.append(object)
            continue

        try:
            input = tf.convert_to_tensor(np.expand_dims(
                masked_img, axis=0), dtype=tf.float32)
            restored_img = restoration_model(input)
            restored_img = tf.squeeze(
                tf.squeeze(restored_img, axis=-1), axis=0)
            ok, buffer = cv2.imencode(
                ".png", (restored_img * 255).numpy().astype('uint8'))
            if ok:
                restored_str = b64encode(buffer).decode()
                object['images']['restored'] = restored_str
        except Exception as e:
            logger.error(e)
            object['probabilities'] = []
            object['error'] = 'restoration_failed'
            objects.append(object)
            continue

        try:
            anomaly_map = blurred * tf.abs(original - restored_img)
            plt.imsave('anomaly.png', anomaly_map, cmap='turbo', vmax=0.7)
            anomaly_str = b64encode(open('anomaly.png', 'rb').read()).decode()
            object['images']['anomaly'] = anomaly_str
            os.remove('anomaly.png')
        except Exception as e:
            logger.error(e)
            object['probabilities'] = []
            object['error'] = 'anomaly_map_failed'
            objects.append(object)
            continue

        try:
            results, explain_image = classifier.predict_proba(
                original_classification, age, max_weight, bmi)

            probabilities = results['kl_grade']
            object['probabilities'] = probabilities
            object['results'] = results
            object['images']['explanation'] = explain_image
            object['error'] = None
        except Exception as e:
            logger.error(e)
            object['probabilities'] = []
            object['results'] = {}
            object['error'] = 'classification_failed'
            objects.append(object)
            continue

        objects.append(object)

    logger.info('Request processed successfully!')
    return make_response(jsonify({
        'objects': objects,
        'length': len(objects),
        'error': None
    }), 200)


@app.route('/crop', methods=['POST'])
@cross_origin()
def crop():
    # crop all objects in the image
    if 'image' not in request.json:
        return make_response(jsonify({
            'cropped': [],
            'length': 0,
            'error': 'invalid_input_format'
        }), 500)

    try:
        img_bytes = b64decode(request.json['image'])  # get image bytes
        original = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        original_copy = original.copy()

        results = extraction_model(original)
        bboxes = results.pandas().xyxy[0].values.tolist()

        if len(bboxes) == 0:
            logger.info('No object detected!')
            return make_response(jsonify({
                'cropped': [],
                'length': 0,
                'error': 'no_object_detected'
            }), 500)
        coordinates = []
        for object in bboxes:
            coordinates.append([int(object[0]), int(
                object[1]), int(object[2]), int(object[3])])
        print(len(coordinates))
    except Exception as e:
        logger.error(e)
        return make_response(jsonify({
            'cropped': [],
            'length': 0,
            'error': 'invalid_input_format'
        }), 500)

    cropped = []
    for i in range(len(coordinates)):
        try:
            x1, y1, x2, y2 = coordinates[i][0:4]
            cropped_image = original[y1:y2, x1:x2, ...]
            cropped_image = cv2.resize(cropped_image, (224, 224))

            cropped_str = b64encode(cv2.imencode(
                '.png', cropped_image)[1]).decode()
            cropped.append(cropped_str)
        except Exception as e:
            logger.error(e)
            cropped.append(None)
            continue

    logger.info('Request processed successfully!')
    return make_response(jsonify({
        'cropped': cropped,
        'length': len(cropped),
        'error': None
    }), 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
