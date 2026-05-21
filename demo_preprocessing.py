# demo_full_pipeline.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_image, get_roi_mask, apply_mask
from feature_extraction import hsv_hist_feature
from predict import predict_image


def demo_full_pipeline(image_path: str):
    # ===== 0. ĐỌC ẢNH GỐC =====
    img_rgb = load_image(image_path)   # ảnh RGB
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # ===== 1. PHÂN ĐOẠN + PHÁT HIỆN BIÊN + HÌNH THÁI HỌC =====
    # get_roi_mask đã làm: phát hiện biên + khép kín + tạo mask
    mask, edges, closed = get_roi_mask(img_rgb)

    # Phân đoạn: dùng mask để tách ROI
    roi_rgb = apply_mask(img_rgb, mask)

    # ===== 2. TRÍCH ĐẶC TRƯNG MÀU HSV (32 chiều) =====
    # (dùng đúng hàm mà bạn train)
    feature_vec = hsv_hist_feature(roi_rgb, mask)   # vector 32 chiều

    # chuyển sang HSV để vẽ minh họa
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # tính lại histogram để vẽ
    hist_h = cv2.calcHist([h], [0], mask, [16], [0, 180])
    hist_s = cv2.calcHist([s], [0], mask, [8],  [0, 256])
    hist_v = cv2.calcHist([v], [0], mask, [8],  [0, 256])

    hist_h = hist_h / (hist_h.sum() + 1e-6)
    hist_s = hist_s / (hist_s.sum() + 1e-6)
    hist_v = hist_v / (hist_v.sum() + 1e-6)

    # ===== 3. NHẬN DẠNG (SVM) =====
    label = predict_image(image_path)

    print("====================================")
    print("Ảnh demo:", image_path)
    print("Nhãn dự đoán (SVM):", label)
    print("Độ dài vector đặc trưng:", len(feature_vec))
    print("Vector 32 chiều (một số giá trị đầu):")
    print(feature_vec[:10])
    print("====================================")

    # ===== 4. VẼ TỪNG KHỐI =====
    fig, axs = plt.subplots(3, 4, figsize=(14, 9))

    # --- Hàng 1: Phân đoạn + biên + hình thái học ---
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title("Ảnh gốc")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(img_gray, cmap="gray")
    axs[0, 1].set_title("Ảnh xám (tiền xử lý)")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(edges, cmap="gray")
    axs[0, 2].set_title("Phát hiện biên\n(Edge detection)")
    axs[0, 2].axis("off")

    axs[0, 3].imshow(closed, cmap="gray")
    axs[0, 3].set_title("Hình thái học\n(Morphology: khép kín)")
    axs[0, 3].axis("off")

    # --- Hàng 2: Phân đoạn (mask + ROI) + kênh HSV ---
    axs[1, 0].imshow(mask, cmap="gray")
    axs[1, 0].set_title("Phân đoạn: Mặt nạ ROI")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(roi_rgb)
    axs[1, 1].set_title("Ảnh ROI (đã tách nền)")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(h, cmap="gray")
    axs[1, 2].set_title("Kênh H (Hue)")
    axs[1, 2].axis("off")

    axs[1, 3].imshow(s, cmap="gray")
    axs[1, 3].set_title("Kênh S (Sat.)")
    axs[1, 3].axis("off")

    # --- Hàng 3: Histogram + nhận dạng ---
    axs[2, 0].plot(hist_h)
    axs[2, 0].set_title("Histogram H (16 bin)")
    axs[2, 0].set_xlabel("Bin")
    axs[2, 0].set_ylabel("Tần suất")

    axs[2, 1].plot(hist_s)
    axs[2, 1].set_title("Histogram S (8 bin)")
    axs[2, 1].set_xlabel("Bin")
    axs[2, 1].set_ylabel("Tần suất")

    axs[2, 2].plot(hist_v)
    axs[2, 2].set_title("Histogram V (8 bin)")
    axs[2, 2].set_xlabel("Bin")
    axs[2, 2].set_ylabel("Tần suất")

    axs[2, 3].axis("off")
    text = (
        "Trích chọn đặc trưng:\n"
        "- Ghép 16(H) + 8(S) + 8(V)\n"
        f"→ Vector 32 chiều\n\n"
        f"Nhận dạng (SVM):\n"
        f"→ Dự đoán: {label}"
    )
    axs[2, 3].text(0.0, 0.5, text, fontsize=10, va="center")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = input("Nhập đường dẫn ảnh (vd: dataset/cat/cat1.jpg): ").strip()
    if not path:
        print("Chưa nhập ảnh.")
    else:
        demo_full_pipeline(path)
