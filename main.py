# main.py

import argparse
from train import train_model
from predict import predict_single_image

def main():
    parser = argparse.ArgumentParser(description="Animal classifier (cat/bird/fish)")
    subparsers = parser.add_subparsers(dest="command")

    # train
    subparsers.add_parser("train", help="Huấn luyện model")

    # predict
    pred_parser = subparsers.add_parser("predict", help="Dự đoán 1 ảnh")
    pred_parser.add_argument("--image", "-i", required=True, help="Đường dẫn tới ảnh")

    args = parser.parse_args()

    if args.command == "train":
        train_model()
    elif args.command == "predict":
        label = predict_single_image(args.image)
        print(f"Kết quả dự đoán: {label}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
