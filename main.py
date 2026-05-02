"""
Sudoku Solver — Computer Vision Pipeline

Solve mode (local)  : Canny detection + Template Matching → interact with browser.
Solve mode (Docker) : Canny detection + Template Matching → render solution onto screenshot image.
Benchmark           : runs all 4 pipeline combinations; --runs N repeats over N fresh puzzles.

Usage
-----
    python main.py
    python main.py --benchmark
    python main.py --benchmark --runs 5

Docker (headless, no display):
    docker run --rm sudoku-solver python main.py
    # Produces solved_sudoku.png in the working directory.
"""

import argparse
import os
import sys

import capture
import solver
import gridDetection
import patternMatch
import benchmark as bm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent benchmark runs (default: 1).")
    return parser.parse_args()


def main():
    args = parse_args()
    is_docker = bool(os.environ.get("DOCKER"))

    url             = "https://sudoku.com/"
    screenshot_path = "sudoku_screenshot.png"

    print(f"\n[1] Opening {url} ...")
    driver = capture.open_browser_and_screenshot(url, screenshot_path)

    # ── Benchmark mode ────────────────────────────────────────────────────────
    if args.benchmark:
        n = args.runs
        if n == 1:
            sys.argv = [sys.argv[0], "--screenshot", screenshot_path]
            bm.main()
        else:
            all_results = []
            for i in range(n):
                print(f"\n{'#'*58}")
                print(f"  RUN {i+1} / {n}")
                print(f"{'#'*58}")
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

    # Also get the raw bbox for rendering (needed in Docker mode)
    import cv2
    import numpy as np
    from patternMatch import order_points
    img_raw  = cv2.imread(screenshot_path)
    gray     = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    blur     = cv2.GaussianBlur(gray, (7, 7), 0)
    edges    = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest  = max(contours, key=cv2.contourArea)
    epsilon  = 0.02 * cv2.arcLength(largest, True)
    approx   = cv2.approxPolyDP(largest, epsilon, True)
    pts      = order_points(approx.reshape(4, 2))
    xs, ys   = pts[:, 0], pts[:, 1]
    grid_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    print("\n[3] Reading digits (Template Matching) ...")
    original_board = patternMatch.read_digits(warped)

    print("\n[4] Solving ...")
    solved_board = solver.solve_sudoku(original_board)
    if solved_board is None:
        print("ERROR: Could not solve — a digit may have been misread.")
        driver.quit()
        sys.exit(1)

    if is_docker:
        # ── Docker mode: render solution onto screenshot ───────────────────
        print("\n[5] Rendering solution onto image ...")
        from render_solution import render_solution
        render_solution(
            screenshot_path = screenshot_path,
            output_path     = "solved_sudoku.png",
            original_board  = original_board,
            solved_board    = solved_board,
            grid_bbox       = grid_bbox,
        )
        print("\nDone. Output: solved_sudoku.png")
    else:
        # ── Local mode: type solution into the browser ─────────────────────
        import interact
        print("\n[5] Entering solution ...")
        _, start_coord = gridDetection.detect_canny(screenshot_path)
        interact.enter_solution(driver, solved_board, original_board, start_coord)
        print("\nDone! Press Enter to close the browser.")
        input()

    driver.quit()


if __name__ == "__main__":
    main()