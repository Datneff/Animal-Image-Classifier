# dataset_loader.py

import os
import glob
import numpy as np
from config import DATASET_DIR, CLASSES
from feature_extraction import extract_feature_from_path

def load_dataset():
    X = []
    y = []

    for label in CLASSES:
        folder = os.path.join(DATASET_DIR, label)
        pattern_jpg = os.path.join(folder, "*.jpg")
        pattern_png = os.path.join(folder, "*.png")

        for img_path in glob.glob(pattern_jpg) + glob.glob(pattern_png):
            try:
                feature = extract_feature_from_path(img_path)
                X.append(feature)
                y.append(label)
                print(f"[OK] {img_path}")
            except Exception as e:
                print(f"[ERR] {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    return X, y
