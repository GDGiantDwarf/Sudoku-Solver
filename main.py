"""
Sudoku Solver — Computer Vision Pipeline

Local mode  : Canny + Template Matching → interact with live browser.
Docker mode : Canny + Template Matching → render solution onto screenshot image.
              Benchmark and YOLO are unavailable in Docker (no ultralytics installed).

Usage
-----
    python main.py
    python main.py --benchmark
    python main.py --benchmark --runs 5

Docker:
    docker run --rm -v $(pwd):/output sudoku-solver \
        bash -c "python main.py && cp solved_sudoku.png /output/"
"""

import argparse
import os
import sys

import cv2

import capture
import gridDetection
import patternMatch
import solver
from patternMatch import order_points
from render_solution import render_solution

IS_DOCKER = bool(os.environ.get("DOCKER"))


def parse_args():
    parser = argparse.ArgumentParser()
    if not IS_DOCKER:
        parser.add_argument("--benchmark", action="store_true")
        parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def _get_grid_bbox(screenshot_path):
    """Re-detect grid corners to get the pixel bbox for rendering."""
    img  = cv2.imread(screenshot_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)
    pts     = order_points(approx.reshape(4, 2))
    xs, ys  = pts[:, 0], pts[:, 1]
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def main():
    args = parse_args()

    url             = "https://sudoku.com/"
    screenshot_path = "sudoku_screenshot.png"

    print(f"\n[1] Opening {url} ...")
    driver = capture.open_browser_and_screenshot(url, screenshot_path)

    # ── Benchmark mode (local only) ───────────────────────────────────────────
    if not IS_DOCKER and getattr(args, "benchmark", False):
        import benchmark as bm
        n = args.runs
        if n == 1:
            sys.argv = [sys.argv[0], "--screenshot", screenshot_path]
            bm.main()
        else:
            all_results = []
            for i in range(n):
                print(f"\n{'#'*58}\n  RUN {i+1} / {n}\n{'#'*58}")
                if i > 0:
                    capture.navigate_and_screenshot(driver, url, screenshot_path)
                run = bm.run_once(screenshot_path, verbose=False,
                                  img_suffix=f"_run{i+1:02d}")
                all_results.append(run)
            bm.print_compound_summary(all_results, n)
        driver.quit()
        return

    # ── Solve mode ────────────────────────────────────────────────────────────
    print("\n[2] Detecting grid (Canny) ...")
    warped, _ = gridDetection.detect_canny(screenshot_path)

    print("\n[3] Reading digits (Template Matching) ...")
    original_board = patternMatch.read_digits(warped)

    print("\n[4] Solving ...")
    solved_board = solver.solve_sudoku(original_board)
    if solved_board is None:
        print("ERROR: Could not solve — a digit may have been misread.")
        driver.quit()
        sys.exit(1)

    if IS_DOCKER:
        # ── Docker: render solution onto screenshot ────────────────────────
        print("\n[5] Rendering solution onto image ...")
        grid_bbox = _get_grid_bbox(screenshot_path)
        render_solution(
            screenshot_path = screenshot_path,
            output_path     = "solved_sudoku.png",
            original_board  = original_board,
            solved_board    = solved_board,
            grid_bbox       = grid_bbox,
        )
        print("\nDone. Output: solved_sudoku.png")
    else:
        # ── Local: type solution into the live browser ─────────────────────
        import interact
        print("\n[5] Entering solution ...")
        _, start_coord = gridDetection.detect_canny(screenshot_path)
        interact.enter_solution(driver, solved_board, original_board, start_coord)
        print("\nDone! Press Enter to close the browser.")
        input()

    driver.quit()


if __name__ == "__main__":
    main()