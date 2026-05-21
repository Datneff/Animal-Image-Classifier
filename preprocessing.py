# preprocessing.py

import cv2
import numpy as np

def load_image(path, target_size=(256, 256)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

def get_roi_mask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Phát hiện biên
    edges = cv2.Canny(blur, 50, 150)

    # 2. Khép kín các đoạn biên
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Tìm contour lớn nhất và fill
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask, edges, closed

def apply_mask(image_rgb, mask):
    masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return masked
