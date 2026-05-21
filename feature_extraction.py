# feature_extraction.py

import cv2
import numpy as np
from preprocessing import load_image, get_roi_mask, apply_mask


def hsv_hist_feature(image_rgb, mask):
    # RGB -> BGR -> HSV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # H:16, S:8, V:8 => 32 chiều
    hist_h = cv2.calcHist([h], [0], mask, [16], [0, 180])
    hist_s = cv2.calcHist([s], [0], mask, [8],  [0, 256])
    hist_v = cv2.calcHist([v], [0], mask, [8],  [0, 256])

    feature = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

    # chuẩn hóa
    if feature.sum() > 0:
        feature = feature / feature.sum()

    return feature.astype(np.float32)


def extract_feature_from_path(image_path):
    # Đọc ảnh + tạo ROI + trích histogram HSV 32 chiều
    img = load_image(image_path)          # trả về ảnh RGB
    mask, edges, closed = get_roi_mask(img)
    masked_img = apply_mask(img, mask)
    feature = hsv_hist_feature(masked_img, mask)
    return feature
