"""
render_solution.py — Docker output mode.

Instead of typing into a live browser (which requires a physical display),
this module draws the solved digits directly onto the original screenshot
and saves the result as an image.

Used automatically when the DOCKER environment variable is set.
"""

import cv2
import numpy as np


def render_solution(
    screenshot_path: str,
    output_path: str,
    original_board: list,
    solved_board: list,
    grid_bbox: tuple,
):
    """
    Overlay solved digits onto the screenshot and save the result.

    Only cells that were empty in the original puzzle (original_board[r][c] == 0)
    receive a new digit — pre-filled cells are left untouched.

    Args:
        screenshot_path : path to the original screenshot PNG
        output_path     : where to write the annotated image
        original_board  : 9x9 list of ints — 0 for empty cells
        solved_board    : 9x9 list of ints — complete solution
        grid_bbox       : (x1, y1, x2, y2) pixel coordinates of the grid
                          in the original screenshot (from gridDetection)
    """
    img = cv2.imread(screenshot_path)
    if img is None:
        raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

    x1, y1, x2, y2 = grid_bbox
    grid_w = x2 - x1
    grid_h = y2 - y1
    cell_w = grid_w / 9
    cell_h = grid_h / 9

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_h / 55          # scale to ~80% of cell height
    thickness  = max(2, int(cell_h / 22))
    color      = (34, 139, 34)        # forest green — distinct from pre-filled navy digits

    for r in range(9):
        for c in range(9):
            if original_board[r][c] != 0:
                continue  # pre-filled — skip

            digit = solved_board[r][c]
            if digit == 0:
                continue  # unsolved cell (shouldn't happen on a valid puzzle)

            # Centre of this cell in screenshot coordinates
            cx = int(x1 + (c + 0.5) * cell_w)
            cy = int(y1 + (r + 0.5) * cell_h)

            text = str(digit)
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            origin = (cx - tw // 2, cy + th // 2)
            cv2.putText(img, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"  Solution image saved: {output_path}")