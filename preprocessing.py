import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import exposure

import matplotlib
matplotlib.rcParams['savefig.pad_inches'] = 0
matplotlib.use('Agg')

st.set_option('deprecation.showPyplotGlobalUse', False)

import ultralytics
from ultralytics import YOLO

IMAGE_SIZE = 224
NUM_CLASSES = 3

yolo_weight = './weights_yolo/oai_s_best4.pt'
seg_model = YOLO(yolo_weight)

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


def find_boundaries(mask, start, end, top=True, verbose=0):
    #     nếu top = True, tìm đường bao bên trên cùng từ left đến right
    #     nếu top = False, tìm đường bao dưới cùng từ left đến right
    contours, _ = cv2.findContours(255 * mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    contour = contours[areas.argmax()]
    contour = contour.reshape(-1, 2)
    org_contour = contour.copy()

    start_idx = ((start - contour) ** 2).sum(axis=-1).argmin()
    end_idx = ((end - contour) ** 2).sum(axis=-1).argmin()
    if start_idx <= end_idx:
        contour = contour[start_idx:end_idx + 1]
    else:
        contour = np.concatenate([contour[start_idx:], contour[:end_idx + 1]])

    if top:
        sorted_indices = np.argsort(contour[:, 1])[::-1]
    else:
        sorted_indices = np.argsort(contour[:, 1])
    contour = contour[sorted_indices]

    unique_indices = sorted(np.unique(contour[:, 0], return_index=True)[1])
    contour = contour[unique_indices]
    sorted_indices = np.argsort(contour[:, 0])
    contour = contour[sorted_indices]
    if verbose:
        temp = draw_points(127 * mask.astype(np.uint8), contour, thickness=5)
        temp = draw_points(temp, [start, end], color=[155, 155], thickness=15)
        cv2_imshow(temp)

    return np.array(contour), np.array(org_contour)


def get_contours(mask, verbose=0):
    limit_points = detect_limit_points(mask, verbose=verbose)
    upper_contour, full_upper = find_boundaries(mask == 1, limit_points[0], limit_points[1], top=False, verbose=verbose)
    lower_contour, full_lower = find_boundaries(mask == 2, limit_points[3], limit_points[2], top=True, verbose=verbose)
    if verbose:
        temp = draw_points(127 * mask, full_upper, thickness=3, color=(255, 0, 0))
        temp = draw_points(temp, full_lower, thickness=3)
        cv2_imshow(temp)
        cv2.imwrite('full.png', temp)
        temp = draw_points(temp, limit_points, thickness=7, color=(0, 0, 255))
        cv2_imshow(temp)
        cv2.imwrite('limit_points.png', temp)
    if verbose:
        temp = draw_points(127 * mask, upper_contour, thickness=3, color=(255, 0, 0))
        temp = draw_points(temp, lower_contour, thickness=3)
        cv2_imshow(temp)
        cv2.imwrite('cropped.png', temp)

    return upper_contour, lower_contour


def cv2_imshow(images):
    if not isinstance(images, list):
        images = [images]

    num_images = len(images)

    # Hiển thị ảnh đơn lẻ trực tiếp bằng imshow
    if num_images == 1:
        image = images[0]
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Ảnh màu (RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
        else:
            # Ảnh xám
            plt.imshow(image, cmap='gray')

        plt.axis("off")
        plt.show()
    else:
        # Hiển thị nhiều ảnh trên cùng một cột
        fig, ax = plt.subplots(num_images, 1, figsize=(4, 4 * num_images))

        for i in range(num_images):
            image = images[i]
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Ảnh màu (RGB)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax[i].imshow(image_rgb)
            else:
                # Ảnh xám
                ax[i].imshow(image, cmap='gray')

            ax[i].axis("off")

        plt.tight_layout()
        plt.show()


def to_color(image):
    if len(image.shape) == 3 and image.shape[-1] == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def to_gray(image):
    if len(image.shape) == 3 and image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the grayscale image
    equalized_image = clahe.apply(gray_image)

    return equalized_image


def detect_edge(image, minVal=100, maxVal=200, blur_size=(5, 5)):
    image_gray = to_gray(image)

    blurred_image = cv2.GaussianBlur(image_gray, blur_size, 0)

    # Phát hiện biên cạnh bằng thuật toán Canny
    edges = cv2.Canny(blurred_image, minVal, maxVal)

    return edges


def show_mask2(image, mask, label2color={1: (255, 255, 0), 2: (0, 255, 255)}, alpha=0.1):
    # Tạo hình ảnh mask từ mask và bảng ánh xạ màu
    image = to_color(image)
    mask_image = np.zeros_like(image)
    for label, color in label2color.items():
        mask_image[mask == label] = color

    mask_image = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)

    # Hiển thị hình ảnh và mask
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask_image)
    ax[1].set_title("Mask")
    ax[1].axis("off")

    plt.show()


def combine_mask(image, mask, label2color={1: (255, 255, 0), 2: (0, 255, 255)}, alpha=0.1):
    image = to_color(image)
    mask_image = np.zeros_like(image)
    for label, color in label2color.items():
        mask_image[mask == label] = color

    mask_image = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
    return mask_image


def draw_points(image, points, color=None, random_color=False, same=True, thickness=1):
    if color is None and not random_color:
        color = (0, 255, 0)  # Màu mặc định là xanh lá cây (BGR)
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    image = to_color(image)

    for point in points:
        if random_color and not same:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        x, y = point
        image = cv2.circle(image, (x, y), thickness, color, -1)  # Vẽ điểm lên ảnh
    return image


def draw_lines(image, pairs, color=None, random_color=False, same=True, thickness=1):
    image_with_line = to_color(np.copy(image))

    if color is None and not random_color:
        color = (0, 255, 0)  # Màu mặc định là xanh lá cây (BGR)
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Vẽ đường thẳng dựa trên danh sách các cặp điểm
    for pair in pairs:

        if random_color and not same:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        start_point = pair[0]
        end_point = pair[1]
        image_with_line = cv2.line(image_with_line, start_point, end_point, color, thickness)
        image_with_line = cv2.circle(image_with_line, start_point, thickness + 1, color, -1)
        image_with_line = cv2.circle(image_with_line, end_point, thickness + 1, color, -1)

    return image_with_line


def detect_limit_points(mask, verbose=0):
    # tìm giới hạn hai bên của khớp gối
    h, w = mask.shape
    res = []
    upper_pivot = np.array([0, w // 2])  # r c
    lower_pivot = np.array([h, w // 2])  # r c

    left_slice = slice(0, w // 2)
    right_slice = slice(w // 2, None)
    center_slice = slice(int(0.2 * h), int(0.8 * h))

    left = np.zeros_like(mask)
    left[center_slice, left_slice] = mask[center_slice, left_slice]

    right = np.zeros_like(mask)
    right[center_slice, right_slice] = mask[center_slice, right_slice]

    if verbose:
        cv2_imshow([left, right])

    pivot = np.array([0, w])
    coords = np.argwhere(left == 1)
    distances = ((coords - pivot) ** 2).sum(axis=-1)
    point = coords[distances.argmax()][::-1]
    res.append(point)

    pivot = np.array([0, 0])
    coords = np.argwhere(right == 1)
    distances = ((coords - pivot) ** 2).sum(axis=-1)
    point = coords[distances.argmax()][::-1]
    res.append(point)

    pivot = np.array([h, w])
    coords = np.argwhere(left == 2)
    distances = ((coords - pivot) ** 2).sum(axis=-1)
    point = coords[distances.argmax()][::-1]
    res.append(point)

    pivot = np.array([h, 0])
    coords = np.argwhere(right == 2)
    distances = ((coords - pivot) ** 2).sum(axis=-1)
    point = coords[distances.argmax()][::-1]
    res.append(point)

    if verbose:
        cv2_imshow(draw_points(127 * mask, res))

    return res


def center(contour):
    #     array = contour[:,1]
    #     min_value = np.min(array)
    #     argmax_indices = np.argwhere(array == min_value)
    #     if len(argmax_indices) == 1:
    #         i = argmax_indices[0]
    #     else:
    #         i = int(np.median(argmax_indices))
    #     return contour[i]
    idx = len(contour) // 2
    return contour[idx]


def pooling_array(array, n, mode='mean'):
    if mode == 'mean':
        pool = lambda x: np.mean(x)
    elif mode == 'min':
        pool = lambda x: np.min(x)
    elif mode == 'sum':
        pool = lambda x: np.sum(x)

    if n == 1:
        return pool(array)

    array_length = len(array)
    if array_length < n:
        return array
    segment_length = array_length // n
    remaining_elements = array_length % n

    if remaining_elements == 0:
        segments = np.split(array, n)
    else:
        mid = remaining_elements * (segment_length + 1)
        segments = np.split(array[:mid], remaining_elements)
        segments += np.split(array[mid:], n - remaining_elements)

    segments = [pool(segment) for segment in segments]

    return np.array(segments)


def distance(mask, upper_contour, lower_contour, p=0.12, verbose=0):
    x_center = (center(lower_contour)[0] + center(upper_contour)[0]) // 2
    length = (lower_contour[-1, 0] - lower_contour[0, 0] + upper_contour[-1, 0] - upper_contour[0, 0]) / 2
    crop_length = int(p * length)
    left = x_center - crop_length // 2
    right = x_center + crop_length // 2
    x_min = max(lower_contour[0, 0], upper_contour[0, 0])
    x_max = min(lower_contour[-1, 0], upper_contour[-1, 0])

    left_idx = np.where(lower_contour[:, 0] == left)[0][0]
    right_idx = np.where(lower_contour[:, 0] == right)[0][0]
    left_lower_contour = lower_contour[left_idx:]
    right_lower_contour = lower_contour[:right_idx + 1][::-1]

    left_lower_contour = lower_contour[(lower_contour[:, 0] <= left) & (lower_contour[:, 0] >= x_min)]
    right_lower_contour = lower_contour[(lower_contour[:, 0] >= right) & (lower_contour[:, 0] <= x_max)][::-1]

    left_upper_contour = upper_contour[(upper_contour[:, 0] <= left) & (upper_contour[:, 0] >= x_min)]
    right_upper_contour = upper_contour[(upper_contour[:, 0] >= right) & (upper_contour[:, 0] <= x_max)][::-1]

    if verbose == 1:
        temp = draw_points(mask * 127, left_lower_contour, color=(0, 255, 0), thickness=3)
        temp = draw_points(temp, right_lower_contour, color=(0, 255, 0), thickness=3)
        temp = draw_points(temp, left_upper_contour, color=(255, 0, 0), thickness=3)
        temp = draw_points(temp, right_upper_contour, color=(255, 0, 0), thickness=3)
        cv2_imshow(temp)
        cv2.imwrite('center_cropped.png', temp)
    links = list(zip(left_upper_contour, left_lower_contour)) + list(zip(right_upper_contour, right_lower_contour))

    temp = left_upper_contour, right_upper_contour, left_lower_contour, right_lower_contour

    return left_lower_contour[:, 1] - left_upper_contour[:, 1], right_lower_contour[:, 1] - right_upper_contour[:,
                                                                                            1], links, temp


# def getMiddle(mask, contour, verbose=0):
#     X = contour[:, 0].reshape(-1, 1)
#     y = contour[:, 1]
#     reg = LinearRegression().fit(X, y)
#     i_min = np.argmin(y[int(len(y) * 0.2):int(len(y) * 0.8)]) + int(len(y) * 0.2)
#     left = i_min - 1
#     right = i_min + 1
#     left_check = False
#     right_check = False
#     if verbose == 1:
#         cmask = draw_points(mask, contour, thickness=2, color=(255, 0, 0))
#         cmask = draw_points(cmask, np.hstack([X, reg.predict(X).reshape(-1, 1).astype('int')]))
#         cv2_imshow(cmask)
#         plt.show()
#     while True:
#         while not left_check:
#             if y[left] > reg.predict(X[left].reshape(-1, 1)):
#                 break
#             left -= 1
#         while not right_check:
#             if y[right] > reg.predict(X[right].reshape(-1, 1)):
#                 break
#             right += 1
#         if verbose == 1:
#             cmask = draw_points(cmask, [contour[left]], thickness=10, color=(255, 255, 0))
#             cmask = draw_points(cmask, [contour[right]], thickness=7, color=(255, 0, 255))
#             cv2_imshow(cmask)
#             plt.show()
#         left_min = np.argmin(y[int(len(y) * 0.2):left]) + int(len(y) * 0.2) if int(len(y) * 0.2) < left else left
#         right_min = np.argmin(y[right:int(len(y) * 0.8)]) + right if right < int(len(y) * 0.8) else right
#         if y[left_min] > reg.predict(X[left_min].reshape(-1, 1)):
#             left_check = True
#         if y[right_min] > reg.predict(X[right_min].reshape(-1, 1)):
#             right_check = True
#         if right_check and left_check:
#             break
#         left = left_min - 1
#         right = right_min + 1
#     return min(X.flatten()[left], X.flatten()[right]), max(X.flatten()[left], X.flatten()[right])


def get_JSW(mask, dim=None, pool='mean', p=0.3, verbose=0):
    if isinstance(mask, str):
        mask = cv2.imread(mask, 0)
    if mask is None:
        return np.zeros(10), np.zeros(10)
    uc, lc = get_contours(mask, verbose=verbose)
    left_distances, right_distances, links, contours = distance(mask, uc, lc, p=p, verbose=verbose)
    if verbose:
        print('in getjsw')
        temp = draw_points(mask * 127, contours[0], thickness=3, color=(255, 0, 0))
        temp = draw_points(temp, contours[1], thickness=3, color=(255, 0, 0))
        temp = draw_points(temp, contours[2], thickness=3, color=(0, 255, 0))
        temp = draw_points(temp, contours[3], thickness=3, color=(0, 255, 0))
        temp = draw_lines(temp, links[::6], color=(0, 0, 255))
        cv2_imshow(temp)
        cv2.imwrite("drawn_lines.png", temp)
    if dim:
        left_distances = pooling_array(left_distances, dim, pool)
        right_distances = pooling_array(right_distances, dim, pool)
    return left_distances, right_distances


def seg(img_path, model=seg_model, verbose=0, combine=False):
    img = cv2.imdecode(np.fromstring(img_path.read(), np.uint8), 1)
    # img = cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    eimg = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eimg = clahe.apply(eimg)
    eimg = to_color(eimg)
    res = seg_model(eimg, verbose=False)
    mask = res[0].masks.data[0] * (res[0].boxes.cls[0] + 1) + res[0].masks.data[1] * (res[0].boxes.cls[1] + 1)
    mask = mask.cpu().numpy()
    if verbose == 1:
        cv2_imshow(eimg)
        cv2.imwrite('original.png', eimg)
        cv2_imshow(combine_mask(eimg, mask))
        plt.show()
    if combine:
        mask = combine_mask(eimg, mask)
    s1 = np.sum(mask == 1)
    s2 = np.sum(mask == 2)

    return mask


def segment(img_path, model=seg_model, verbose=0, combine=True):
    img = cv2.imdecode(np.fromstring(img_path, np.uint8), 1)
    # img = cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    eimg = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eimg = clahe.apply(eimg)
    eimg = to_color(eimg)
    res = seg_model(eimg, verbose=False)
    mask = res[0].masks.data[0] * (res[0].boxes.cls[0] + 1) + res[0].masks.data[1] * (res[0].boxes.cls[1] + 1)
    mask = mask.cpu().numpy()
    if verbose == 1:
        cv2_imshow(eimg)
        cv2.imwrite('original.png', eimg)
        cv2_imshow(combine_mask(eimg, mask))
        plt.show()
    if combine:
        mask = combine_mask(eimg, mask)
    s1 = np.sum(mask == 1)
    s2 = np.sum(mask == 2)

    return mask


def split_img(img):
    img_size = img.shape
    return img[:, :(img_size[1] // 3), :], img[:, (img_size[1] // 3 * 2):, :]


def combine_mask(image, mask, label2color={1: (255, 255, 0), 2: (0, 255, 255)}, alpha=0.1):
    image = to_color(image)
    image = cv2.resize(image, mask.shape)
    mask_image = np.zeros_like(image)
    for label, color in label2color.items():
        mask_image[mask == label] = color

    mask_image = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
    return mask_image


def check_outliers(mask):
    pass


def find_boundaries_v2(mask, top=True, verbose=0):
    boundaries = []
    height, width = mask.shape

    contours, _ = cv2.findContours(255 * mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    contour = contours[areas.argmax()]
    contour = contour.reshape(-1, 2)
    org_contour = contour.copy()
    pos = (contour[:, 1].max() + contour[:, 1].min()) // 2
    idx = np.where(contour[:, 1] == pos)
    if contour[idx[0][0]][0] < contour[idx[0][1]][0] and not top:
        start = contour[idx[0][0]]
        end = contour[idx[0][1]]
    else:
        end = contour[idx[0][0]]
        start = contour[idx[0][1]]
    start_idx = ((start - contour) ** 2).sum(axis=-1).argmin()
    end_idx = ((end - contour) ** 2).sum(axis=-1).argmin()
    if start_idx <= end_idx:
        contour = contour[start_idx:end_idx + 1]
    else:
        contour = np.concatenate([contour[start_idx:], contour[:end_idx + 1]])
    if verbose:
        temp = draw_points(127 * mask.astype(np.uint8), contour, thickness=5)
        temp = draw_points(temp, [start, end], color=[155, 155], thickness=15)
        cv2_imshow(temp)

    return np.array(contour), np.array(org_contour)


def get_contours_v2(mask, verbose=0):
    upper_contour, full_upper = find_boundaries_v2(mask == 1, top=False, verbose=verbose)
    lower_contour, full_lower = find_boundaries_v2(mask == 2, top=True, verbose=verbose)
    if verbose:
        temp = draw_points(127 * mask, full_upper, thickness=3, color=(255, 0, 0))
        temp = draw_points(temp, full_lower, thickness=3)
        plt.imshow(temp)
        plt.axis('off')
        plt.show()
        # st.pyplot()
        temp = draw_points(127 * mask, upper_contour, thickness=3, color=(255, 0, 0))
        temp = draw_points(temp, lower_contour, thickness=3)
        cv2_imshow(temp)
    
    return upper_contour, lower_contour
