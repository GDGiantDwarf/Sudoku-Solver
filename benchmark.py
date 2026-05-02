"""
Benchmark: four pipeline combinations compared on the same screenshot.

Grid detection  x  Digit recognition:
  1. Canny + Template Matching  (used as ground truth for comparison)
  2. Canny + Tesseract
  3. YOLO  + Template Matching
  4. YOLO  + Tesseract

Usage
-----
    python benchmark.py                        # single run
    python benchmark.py --screenshot path.png  # on existing screenshot
"""

import argparse
import sys
import time
import os

import cv2
import numpy as np

import gridDetection
import patternMatch
import ocrProcessing

PIPELINES = [
    ("Canny + Template Matching", gridDetection.detect_canny, patternMatch.read_digits),
    ("Canny + Tesseract",         gridDetection.detect_canny, ocrProcessing.read_digits),
    ("YOLO  + Template Matching", gridDetection.detect_yolo,  patternMatch.read_digits),
    ("YOLO  + Tesseract",         gridDetection.detect_yolo,  ocrProcessing.read_digits),
]

GROUND_TRUTH_PIPELINE = "Canny + Template Matching"
PIPELINE_NAMES        = [p[0] for p in PIPELINES]
SHORT_NAMES           = ["Canny+TM", "Canny+OCR", "YOLO+TM", "YOLO+OCR"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name):
    return name.lower().replace(" ", "_").replace("+", "").replace("__", "_").strip("_")


def print_grid(board, name):
    print(f"\n  {name}:")
    for r in range(9):
        if r % 3 == 0:
            print("  +-------+-------+-------+")
        row = "  |"
        for c in range(9):
            row += f" {board[r][c] if board[r][c] else '.'}"
            if c % 3 == 2:
                row += " |"
        print(row)
    print("  +-------+-------+-------+")


def save_debug_image(warped_gray, board, gt_board, name, suffix=""):
    size = warped_gray.shape[0]
    cell = size // 9
    img  = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
    for i in range(10):
        p = i * cell
        t = 3 if i % 3 == 0 else 1
        cv2.line(img, (p, 0), (p, size), (80, 80, 80), t)
        cv2.line(img, (0, p), (size, p), (80, 80, 80), t)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = cell / 60
    th   = max(1, int(cell / 30))
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == 0:
                continue
            gt = gt_board[r][c] if gt_board else val
            if gt_board is None or gt == 0:
                color = (200, 140, 0)
            elif val == gt:
                color = (0, 180, 0)
            else:
                color = (0, 0, 220)
            cx = int((c + 0.5) * cell)
            cy = int((r + 0.5) * cell)
            text = str(val)
            (tw, th2), _ = cv2.getTextSize(text, font, fs, th)
            cv2.putText(img, text, (cx - tw // 2, cy + th2 // 2),
                        font, fs, color, th, cv2.LINE_AA)
    script_dir = os.path.dirname(os.path.abspath(__file__)) + "\\debug_images"
    path = os.path.join(script_dir, f"{_slug(name)}{suffix}.png")
    os.makedirs(script_dir, exist_ok=True)  
    cv2.imwrite(path, img)
    print(f"  Debug image : {path}")


def accuracy_vs_gt(board, gt_board):
    correct = sum(1 for r in range(9) for c in range(9)
                  if board[r][c] == gt_board[r][c])
    return correct, 81, 100 * correct / 81


def agreement_rate(board_a, board_b):
    agree = total = 0
    for r in range(9):
        for c in range(9):
            a, b = board_a[r][c], board_b[r][c]
            if a == 0 and b == 0:
                continue
            total += 1
            if a == b:
                agree += 1
    return (100 * agree / total) if total else 0.0


# ---------------------------------------------------------------------------
# Single-run core
# ---------------------------------------------------------------------------

def run_once(screenshot_path, verbose=True, img_suffix=""):
    """
    Run all 4 pipelines on one screenshot.
    Returns a dict: {pipeline_name: {board, elapsed, correct, warped, error}}
    verbose=True  prints grids and saves debug images.
    verbose=False prints only one-line results per pipeline.
    """
    results = {}

    for name, detect_fn, read_fn in PIPELINES:
        if verbose:
            print(f"\n{'='*58}\n  {name}\n{'='*58}")
        else:
            print(f"  {name:<32} ", end="", flush=True)

        try:
            t0            = time.perf_counter()
            warped, coord = detect_fn(screenshot_path)
            t_detect      = time.perf_counter() - t0
            t1            = time.perf_counter()
            board         = read_fn(warped)
            t_read        = time.perf_counter() - t1
            elapsed       = t_detect + t_read
            filled        = sum(1 for row in board for v in row if v != 0)

            if verbose:
                print(f"  Detection : {t_detect:.2f} s")
                print(f"  Reading   : {t_read:.2f} s")
                print(f"  Total     : {elapsed:.2f} s   |   Cells found: {filled}/81")
                print_grid(board, name)
            else:
                print(f"{elapsed:.2f} s   cells: {filled}/81")

            results[name] = {"board": board, "coord": coord, "warped": warped,
                             "elapsed": elapsed, "error": None}

        except Exception as e:
            if verbose:
                import traceback; traceback.print_exc()
            else:
                print(f"ERROR: {e}")
            results[name] = {"board": [[0]*9 for _ in range(9)], "coord": (0,0),
                             "warped": None, "elapsed": 0.0, "error": str(e)}

    # Compute accuracy vs GT for this run
    gt_board = results[GROUND_TRUTH_PIPELINE]["board"]
    for name, data in results.items():
        if name == GROUND_TRUTH_PIPELINE or data["error"]:
            data["correct"] = None
        else:
            c, _, _ = accuracy_vs_gt(data["board"], gt_board)
            data["correct"] = c

    print(f"\n--- Saving debug images ---")
    for name, data in results.items():
        if data["warped"] is not None:
            gt = None if name == GROUND_TRUTH_PIPELINE else gt_board
            save_debug_image(data["warped"], data["board"], gt, name, img_suffix)

    return results


# ---------------------------------------------------------------------------
# Single-run reporting
# ---------------------------------------------------------------------------


def print_summary(results):
    print(f"\n{'='*72}")
    print("  BENCHMARK RESULTS  (ground truth: Canny + Template Matching)")
    print(f"{'='*72}")
    print(f"  {'Pipeline':<30} {'Time (s)':>8}  {'Cells':>7}  {'Accuracy':>12}")
    print(f"  {'-'*65}")
    for name, data in results.items():
        filled  = sum(1 for row in data["board"] for v in row if v != 0)
        elapsed = data["elapsed"]
        if name == GROUND_TRUTH_PIPELINE:
            acc_str = "  (reference)"
        elif data["error"]:
            acc_str = "  [FAILED]"
        else:
            pct = 100 * data["correct"] / 81
            acc_str = f"  {data['correct']:>4}/81 ({pct:.1f}%)"
        print(f"  {name:<30} {elapsed:>8.2f}  {filled:>5}/81{acc_str}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Multi-run compound reporting
# ---------------------------------------------------------------------------

def print_compound_summary(all_results, n_runs):
    """Aggregate stats across n_runs independent runs."""
    print(f"\n{'='*72}")
    print(f"  COMPOUND RESULTS  ({n_runs} runs, ground truth: Canny + Template Matching)")
    print(f"{'='*72}")
    print(f"  {'Pipeline':<30} {'Avg time':>9}  {'Total acc':>10}  {'Per-run acc':>12}")
    print(f"  {'-'*68}")

    for name in PIPELINE_NAMES:
        times    = [r[name]["elapsed"] for r in all_results]
        avg_time = sum(times) / n_runs

        if name == GROUND_TRUTH_PIPELINE:
            print(f"  {name:<30} {avg_time:>9.2f}  {'(reference)':>10}")
            continue

        corrects   = [r[name]["correct"] for r in all_results
                      if r[name]["correct"] is not None]
        total_corr = sum(corrects)
        total_pct  = 100 * total_corr / (81 * len(corrects)) if corrects else 0
        per_run    = [100 * c / 81 for c in corrects]
        mn, mx     = min(per_run), max(per_run)
        print(f"  {name:<30} {avg_time:>9.2f}  "
              f"{total_corr:>4}/{81*len(corrects)} ({total_pct:.1f}%)  "
              f"[{mn:.1f}%–{mx:.1f}%]")

    # Compound agreement matrix (mean across runs)
    print(f"\n  Mean agreement matrix ({n_runs} runs):")
    col_w = 11
    print(f"  {'':18}" + "".join(f"{s:>{col_w}}" for s in SHORT_NAMES))
    for i, name_a in enumerate(PIPELINE_NAMES):
        row = f"  {SHORT_NAMES[i]:<18}"
        for j, name_b in enumerate(PIPELINE_NAMES):
            if i == j:
                row += f"{'—':>{col_w}}"
            else:
                rates = [agreement_rate(r[name_a]["board"], r[name_b]["board"])
                         for r in all_results]
                row += f"{sum(rates)/len(rates):>{col_w-1}.1f}%"
        print(row)
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", default="sudoku_screenshot.png")
    parser.add_argument("--yolo-weights", default=gridDetection.YOLO_WEIGHTS)
    args = parser.parse_args()
    gridDetection.YOLO_WEIGHTS = args.yolo_weights

    results = run_once(args.screenshot, verbose=True)
    print_summary(results)


if __name__ == "__main__":
    main()