"""
Digit recognition via Tesseract OCR — second approach.

Uses the same geometric pipeline as patternMatch.py (Canny edges -> largest
contour -> perspective warp -> 9x9 cell grid) but replaces template matching
with Tesseract for digit reading.

Key findings from calibration:
  - Cell resize target: 64x64 px. Larger sizes (128, 256) cause Tesseract to
    split a single digit into multiple segments (e.g. '5' -> 'OD'), returning
    empty output with the digit whitelist active.
  - PSM 6 (uniform block) is more reliable than PSM 10 (single character) for
    the sudoku.com font at this scale.
  - Padding: 40 px white border on all sides.

System requirement
------------------
Tesseract must be installed separately from the Python package:
  - Windows : https://github.com/UB-Mannheim/tesseract/wiki
              Then either add the install folder to PATH, or set:
              set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
  - Linux   : sudo apt install tesseract-ocr
  - macOS   : brew install tesseract
"""

import os
import cv2
import numpy as np
import pytesseract

from patternMatch import order_points, perspective_transform

CELL_MARGIN = 6

# PSM 6: uniform block of text — more tolerant than PSM 10 for sudoku digits.
# Cell size 64 px: avoids over-segmentation of large isolated digits.
# tessedit_char_whitelist is unreliable with the LSTM engine (oem 3) on
# Tesseract 5 / Windows — it silently suppresses all output. Digit
# filtering is done in Python instead (see _recognize_digit_ocr).
_TESS_CONFIG = "--psm 6 --oem 3"
_CELL_SIZE   = 64
_CELL_PAD    = 40

_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


def _recognize_digit_ocr(cell_gray_raw):
    """
    Read a single digit from a raw (non-binarized) grayscale cell image.

    Preprocessing:
      1. Resize to 64x64 with INTER_AREA (shrink-safe interpolation).
      2. Per-cell Otsu binarization — adapts to local brightness so the
         approach works even when the page has a colour overlay or tint.
      3. Add 40 px white padding on all sides (Tesseract needs margin).

    Returns the recognised digit (1-9) or 0 if the cell appears empty or
    Tesseract cannot confidently identify a digit.
    """
    cell_small = cv2.resize(cell_gray_raw, (_CELL_SIZE, _CELL_SIZE),
                            interpolation=cv2.INTER_AREA)

    _, cell_bin = cv2.threshold(cell_small, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    padded = cv2.copyMakeBorder(cell_bin, _CELL_PAD, _CELL_PAD,
                                _CELL_PAD, _CELL_PAD,
                                cv2.BORDER_CONSTANT, value=255)

    text = pytesseract.image_to_string(padded, config=_TESS_CONFIG).strip()

    digits = [c for c in text if c in "123456789"]
    if len(digits) == 1:
        return int(digits[0])
    return 0


def _detect_grid_corners(img):
    """
    Locate the sudoku grid via Canny -> largest contour -> 4-corner approximation.
    Identical logic to patternMatch so both approaches use the same grid boundary.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) != 4:
        raise RuntimeError(
            f"Grid detection failed: expected 4 corners, got {len(approx)}."
        )

    pts = order_points(approx.reshape(4, 2))
    return pts, gray


def extract_board_from_screenshot(path):
    """
    Extract the 9x9 board from a screenshot using Tesseract for digit reading.

    Pipeline:
      1. Load screenshot.
      2. Detect grid corners (Canny + largest contour).
      3. Perspective-warp grid to square top-down view (raw grayscale).
      4. Slice into 81 cells (CELL_MARGIN trim on each side).
      5. Read each cell with Tesseract (PSM 6, 64 px, per-cell Otsu).

    Cells are extracted from the RAW grayscale warp so that per-cell
    binarization adapts to local contrast rather than a global threshold.

    Returns:
        board         -- 9x9 list of ints (0 = empty cell)
        cell00_screen -- (x, y) pixel centre of cell (0,0) in screenshot space
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Screenshot not found: {path}")

    pts, gray = _detect_grid_corners(img)
    warped_raw = perspective_transform(gray, pts)

    tl, tr, _, bl = pts
    right_vec = (tr - tl) / 9.0
    down_vec  = (bl - tl) / 9.0
    cell00_center = tl + 0.5 * right_vec + 0.5 * down_vec
    cell00_screen = (int(round(cell00_center[0])), int(round(cell00_center[1])))

    size = warped_raw.shape[0]
    cell_size = size // 9
    board = [[0] * 9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            y1 = r * cell_size + CELL_MARGIN
            y2 = (r + 1) * cell_size - CELL_MARGIN
            x1 = c * cell_size + CELL_MARGIN
            x2 = (c + 1) * cell_size - CELL_MARGIN
            cell_img = warped_raw[y1:y2, x1:x2]
            board[r][c] = _recognize_digit_ocr(cell_img)

    print("\n[Tesseract] Detected board:")
    for row in board:
        print(row)
    print(f"[Tesseract] Cell (0,0) screen centre: {cell00_screen}")

    return board, cell00_screen

# ---------------------------------------------------------------------------
# Modular digit reader (used by benchmark for grid-detector combinations)
# ---------------------------------------------------------------------------

def read_digits(warped_gray):
    """
    Run Tesseract digit recognition on a pre-detected warped grid.

    Args:
        warped_gray -- grayscale square grid image from gridDetection.py

    Returns:
        board -- 9x9 list of ints (0 = empty)
    """
    size      = warped_gray.shape[0]
    cell_size = size // 9
    board     = [[0] * 9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            y1 = r * cell_size + CELL_MARGIN
            y2 = (r + 1) * cell_size - CELL_MARGIN
            x1 = c * cell_size + CELL_MARGIN
            x2 = (c + 1) * cell_size - CELL_MARGIN
            board[r][c] = _recognize_digit_ocr(warped_gray[y1:y2, x1:x2])

    return board