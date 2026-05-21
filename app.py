# app.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from predict import predict_image   # dùng hàm phía trên


def choose_image():
    path = filedialog.askopenfilename(
        title="Chọn ảnh cần phân loại",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not path:
        return

    try:
        # Hiển thị ảnh thu nhỏ
        img = Image.open(path)
        img.thumbnail((300, 300))
        tk_img = ImageTk.PhotoImage(img)
        image_label.configure(image=tk_img)
        image_label.image = tk_img  # giữ reference

        # Gọi model dự đoán
        label = predict_image(path)
        result_label.config(text=f"Kết quả: {label}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không phân loại được ảnh:\n{e}")


root = tk.Tk()
root.title("Animal Classifier - Cat / Bird / Fish")
root.geometry("500x500")

btn = tk.Button(root, text="Chọn ảnh...", command=choose_image, font=("Segoe UI", 12))
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="Kết quả: (chưa có)", font=("Segoe UI", 14, "bold"))
result_label.pack(pady=10)

root.mainloop()
