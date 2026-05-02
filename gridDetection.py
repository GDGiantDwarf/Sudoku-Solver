"""
Grid detection — two approaches.

Both functions take a screenshot path and return (warped_gray, cell00_screen):
  warped_gray   -- grayscale square crop of the grid, ready for cell slicing
  cell00_screen -- (x, y) pixel centre of cell (0,0) in original screenshot space

detect_canny : classical CV  — Canny edges → largest contour → perspective warp
detect_yolo  : learned model — YOLOv8 bbox → axis-aligned crop → square resize
"""

import cv2
import numpy as np
from ultralytics import YOLO

from patternMatch import order_points, perspective_transform

YOLO_WEIGHTS = "runs/detect/sudoku_detector/weights/best.pt"

_yolo_model = None


def _get_yolo(weights):
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(weights)
    return _yolo_model


# ---------------------------------------------------------------------------

def detect_canny(path):
    """
    Detect the sudoku grid using Canny edge detection.

    Pipeline: grayscale -> GaussianBlur -> Canny -> largest contour ->
              4-corner approx -> perspective warp to square top-down view.
    """
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) != 4:
        raise RuntimeError(
            f"Canny grid detection failed: expected 4 corners, got {len(approx)}."
        )

    pts    = order_points(approx.reshape(4, 2))
    warped = perspective_transform(gray, pts)

    tl, tr, _, bl = pts
    right_vec     = (tr - tl) / 9.0
    down_vec      = (bl - tl) / 9.0
    center        = tl + 0.5 * right_vec + 0.5 * down_vec
    cell00_screen = (int(round(center[0])), int(round(center[1])))

    return warped, cell00_screen


def detect_yolo(path, weights=YOLO_WEIGHTS):
    """
    Detect the sudoku grid using the fine-tuned YOLOv8 model.

    Pipeline: YOLO inference -> highest-confidence bbox -> axis-aligned crop
              -> resize to square.

    Unlike Canny, YOLO does not produce corner points for perspective warping.
    Because the grid is browser-rendered it is always axis-aligned, so a
    simple crop + square resize is equivalent.
    """
    model   = _get_yolo(weights)
    img     = cv2.imread(path)
    results = model(path, verbose=False)
    boxes   = results[0].boxes

    if not boxes or len(boxes) == 0:
        raise RuntimeError("YOLO grid detection failed: no box detected.")

    best        = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])

    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop   = gray[y1:y2, x1:x2]
    size   = max(crop.shape)
    warped = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    cell_w        = (x2 - x1) / 9.0
    cell_h        = (y2 - y1) / 9.0
    cell00_screen = (int(x1 + cell_w / 2), int(y1 + cell_h / 2))

    return warped, cell00_screen