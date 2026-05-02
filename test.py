"""
Visual inspection of the trained YOLO model.

Loads images one by one and overlays the predicted bounding box.
  SPACE  -> next image
  ESC    -> quit

Usage
-----
    python test.py
    python test.py --weights runs/detect/sudoku_detector/weights/best.pt
    python test.py --images  dataset/images/train
"""

import argparse
import os
import cv2
from ultralytics import YOLO


def run(weights, img_dir):
    model = YOLO(weights)

    files = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not files:
        print(f"No images found in {img_dir}")
        return

    print(f"Loaded {len(files)} images. SPACE = next, ESC = quit.")

    for path in files:
        img = cv2.imread(path)
        results = model(path, verbose=False)
        boxes   = results[0].boxes

        if boxes and len(boxes):
            best = max(boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, best.xyxy[0])
            conf = float(best.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "NO DETECTION", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow(os.path.basename(path), img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 27:  # ESC
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/sudoku_detector/weights/best.pt")
    parser.add_argument("--images",  default="dataset/images/train")
    args = parser.parse_args()
    run(args.weights, args.images)