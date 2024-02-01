import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.rcParams['savefig.pad_inches'] = 0
matplotlib.use('Agg')

from DCGAN import DCGAN, DropBlockNoise
from preprocessing import seg, get_contours_v2, draw_points

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

from scipy import ndimage
from PIL import Image

from skimage import exposure
from skimage.filters import gaussian

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

IMAGE_SIZE = 224
NUM_CLASSES = 3

yolo_weight = './weights_yolo/oai_s_best4.pt'
seg_model = YOLO(yolo_weight)

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

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    # plt.title(title)
    plt.axis('off')
    plt.show()


# đảo màu những ảnh bị ngược màu
def remove_negative(img):
  outside = np.mean(img[ : , 0])
  inside = np.mean(img[ : , int(IMAGE_SIZE / 2)])
  if outside < inside:
    return img
  else:
    return 1 - img
# lựa chọn tiền xử lý: ảnh gốc, Equalization histogram, CLAHE
def preprocess(img):
    img = remove_negative(img)

    img = exposure.equalize_hist(img)
    img = exposure.equalize_adapthist(img)
    img = exposure.equalize_hist(img)
    return img
# dilate contour
def dilate(mask_img):
    kernel_size = 2 * 22 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return ndimage.binary_dilation(mask_img == 0, structure=kernel)

# Tiêu đề của ứng dụng
st.title("Tải và hiển thị ảnh")

# Hiển thị widget tải tệp tin ảnh
uploaded_file = st.file_uploader("Chọn một tệp tin ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc dữ liệu ảnh từ tệp tin tải lên
    mask = seg(uploaded_file)

    # Sử dụng Matplotlib để đọc và hiển thị ảnh C
    img = plt.imread(uploaded_file, 0)
    img = np.array(Image.fromarray(img).resize((224, 224)))
    img = preprocess(img)

    # Hiển thị ảnh gốc
    show_image(img, title="Original image")
    plt.axis('off')
    st.pyplot()


    uc, lc = get_contours_v2(mask, verbose=1)
    # img = cv2.imread(filepath)
    mask = np.zeros((640, 640)).astype('uint8')
    mask = draw_points(mask, lc, thickness=1, color=(255, 255, 255))
    mask = draw_points(mask, uc, thickness=1, color=(255, 255, 255))
    mask = cv2.resize(mask, (224, 224), cv2.INTER_NEAREST)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask / 255.

    show_image(mask, title = "Contour")
    plt.axis('off')
    st.pyplot()
    # sử dụng equalization histogram
    mask = 1 - mask
    dilated = gaussian(dilate(mask), sigma=50, truncate=0.3)

    im = np.expand_dims(img * (1 - dilated), axis=0)
    im = tf.convert_to_tensor(im, dtype=tf.float32)

    restored_img = restore_model(im)

    res = tf.squeeze(tf.squeeze(restored_img, axis=-1), axis=0)

    show_image(im[0], title="Masked Image")
    plt.axis('off')
    st.pyplot()

    show_image(res, title="Reconstructed image")
    plt.axis('off')
    st.pyplot()

    show_image(dilated*tf.abs(img-res), title="Anomaly map", cmap_type='turbo')
    plt.axis('off')
    st.pyplot()


    plt.imshow(img, cmap = 'gray')
    plt.imshow(dilated*tf.abs(img-res), cmap ='turbo', alpha = 0.3)
    plt.axis('off')
    plt.show()
    st.pyplot()
