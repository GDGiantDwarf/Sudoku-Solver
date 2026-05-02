"""
Train a YOLOv8n model to detect the sudoku grid.

Usage
-----
    python train.py
    python train.py --data dataset/data.yaml --epochs 50 --model yolov8n.pt
"""

import argparse
from ultralytics import YOLO


def train(data, model, epochs, imgsz, project, name):
    net = YOLO(model)
    results = net.train(
        data    = data,
        epochs  = epochs,
        imgsz   = imgsz,
        project = project,
        name    = name,
        exist_ok= True,
    )
    print(f"\nWeights saved to: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="dataset/data.yaml")
    parser.add_argument("--model",   default="yolov8n.pt")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name",    default="sudoku_detector")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.project, args.name)