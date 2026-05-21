# train.py (bản nâng cấp)
from pathlib import Path
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from feature_extraction import extract_feature_from_path

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LABELS = ['bird', 'cat', 'fish']


def load_dataset():
    X, y = [], []

    for label_name in LABELS:
        class_dir = DATASET_DIR / label_name
        if not class_dir.exists():
            print(f"[Cảnh báo] Không thấy thư mục: {class_dir}")
            continue

        for img_path in class_dir.glob("*.*"):
            try:
                feat = extract_feature_from_path(str(img_path))
                X.append(feat)
                y.append(LABELS.index(label_name))
            except Exception as e:
                print(f"[Bỏ qua] {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"\nTổng số mẫu: {len(y)}")
    return X, y


def main():
    print("=== Đang load dataset ===")
    X, y = load_dataset()
    if len(y) == 0:
        print("Không có ảnh nào để train.")
        return

    # 1) Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2) Dùng GridSearch để tìm tham số SVM tốt hơn
    print("=== Đang grid-search SVM ===")
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"],
    }

    base_clf = SVC(probability=True)
    grid = GridSearchCV(
        base_clf,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)

    best_clf = grid.best_estimator_

    # 3) Đánh giá trên tập test
    y_pred = best_clf.predict(X_test)
    print("\n=== Kết quả trên tập test ===")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # 4) Lưu model
    model_path = MODEL_DIR / "animal_model.pkl"
    joblib.dump(best_clf, model_path)
    print(f"\nĐã lưu model vào: {model_path}")


if __name__ == "__main__":
    main()
