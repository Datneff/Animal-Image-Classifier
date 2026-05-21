# predict.py
import numpy as np
import joblib

from feature_extraction import extract_feature_from_path

# ĐÚNG tên file model mà train.py vừa tạo
MODEL_PATH = "models/animal_model.pkl"

# 3 lớp tương ứng với folder trong dataset
LABELS = ['bird', 'cat', 'fish']


# Load model 1 lần
model = joblib.load(MODEL_PATH)


def predict_image(image_path: str) -> str:
    # Lấy vector đặc trưng từ đường dẫn ảnh
    feat = extract_feature_from_path(image_path)   # (32,)
    feat = np.array(feat).reshape(1, -1)          # (1, 32) cho model

    pred = model.predict(feat)[0]

    # Nếu model trả index (0,1,2) thì map sang LABELS
    if isinstance(pred, (int, np.integer)):
        return LABELS[pred]

    # Nếu model đã trả sẵn string label
    return str(pred)
