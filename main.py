"""
Sudoku Solver — Computer Vision Pipeline

Solve mode  : YOLO detection + Template Matching (best per benchmark).
Benchmark   : runs all 4 pipeline combinations; --runs N repeats over N fresh puzzles.

Usage
-----
    python main.py
    python main.py --benchmark
    python main.py --benchmark --runs 5
"""

import argparse
import sys

import capture
import interact
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

    url             = "https://sudoku.com/"
    screenshot_path = "sudoku_screenshot.png"

    print(f"\n[1] Opening {url} ...")
    driver = capture.open_browser_and_screenshot(url, screenshot_path)

    if args.benchmark:
        n = args.runs
        if n == 1:
            # Single run — full verbose output
            sys.argv = [sys.argv[0], "--screenshot", screenshot_path]
            bm.main()
        else:
            # Multi-run — brief per-pipeline output, compound summary at the end
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
    print("\n[2] Detecting grid (YOLO) ...")
    warped, start_coord = gridDetection.detect_yolo(screenshot_path)

    print("\n[3] Reading digits (Template Matching) ...")
    original_board = patternMatch.read_digits(warped)

    print("\n[4] Solving ...")
    solved_board = solver.solve_sudoku(original_board)
    if solved_board is None:
        print("ERROR: Could not solve — a digit may have been misread.")
        driver.quit()
        sys.exit(1)

    print("\n[5] Entering solution ...")
    interact.enter_solution(driver, solved_board, original_board, start_coord)

    print("\nDone! Press Enter to close the browser.")
    input()
    driver.quit()


if __name__ == "__main__":
    main()