# Sudoku Solver — Computer Vision

A Sudoku solver driven entirely by **computer vision**: the program takes a screenshot of [sudoku.com](https://sudoku.com/), detects the grid and reads the digits visually, solves the puzzle, and types the solution into the browser — **without reading the HTML DOM** to perceive the grid or its digits (fundamental constraint of the assignment).

---

## Pipeline

| Step | What happens | File |
|------|-------------|------|
| 1 — Screenshot | Selenium opens sudoku.com, dismisses cookie/tutorial popups, takes a full-page screenshot | `capture.py` |
| 2 — Grid detection | Canny edge detection → largest external contour → 4-corner approximation → perspective warp to square top-down view | `patternMatch.py`, `ocrProcessing.py` |
| 3 — Digit recognition | **Approach A**: template matching (OpenCV `matchTemplate`) against real templates extracted from sudoku.com. **Approach B**: EasyOCR neural model on each cell crop | `patternMatch.py`, `ocrProcessing.py` |
| 4 — Solving | `py-sudoku` constraint-propagation engine | `solver.py` |
| 5 — Interaction | Single pyautogui click on cell (0,0) for grid focus, then arrow keys + digit keys to fill every empty cell — no DOM interaction | `interact.py` |

DOM usage is limited exclusively to closing the cookie banner and tutorial modal (exception explicitly permitted by the assignment).

---

## Tech stack

| Brick | Library | Role |
|-------|---------|------|
| Browser automation | `selenium` + ChromeDriver | Open sudoku.com, screenshot, popup dismissal |
| Grid detection | `opencv-python` | Canny, contour detection, perspective transform, binarisation |
| Digit recognition A | `opencv-python` (`cv2.matchTemplate`) | Template matching against real digit images |
| Digit recognition B | `pytesseract` + Tesseract | OCR on individual cell crops (PSM 10, digit whitelist) |
| Solving | `py-sudoku` | Constraint-propagation engine (external library) |
| Input automation | `pyautogui` | Click + arrow keys + digit key presses |

---

## Installation

### Prerequisites

- Python 3.11+
- Google Chrome + matching [ChromeDriver](https://chromedriver.chromium.org/downloads) on your `PATH`
- (Linux only) `libgl1` and `libglib2.0-0` system packages for OpenCV

### Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Tesseract system install** — `pytesseract` is only a Python wrapper; the Tesseract binary must be installed separately:
> - **Windows**: download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), then either add the install folder to your `PATH` or set `TESSERACT_CMD`:
>   ```powershell
>   $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
>   ```
> - **Linux**: `sudo apt install tesseract-ocr`
> - **macOS**: `brew install tesseract`

### Verify ChromeDriver

```bash
chromedriver --version   # must match your Chrome version
```

---

## Running the solver

```bash
# Solve the current puzzle on sudoku.com (uses Template Matching — best method per benchmark)
python main.py

# Run the benchmark instead of solving (compare Template Matching vs EasyOCR)
python main.py --benchmark

# Benchmark with a known ground-truth string for accuracy measurement
python main.py --benchmark --ground-truth 058030020402000905007000680290054070500620000003810025109003064865049130070000006
```

> **Note**: Do not move your mouse during the input phase — `pyautogui` takes full keyboard/mouse control while typing the solution.

---

## Benchmarking both approaches

The benchmark runs **Template Matching** and **EasyOCR** on the same screenshot under identical conditions and reports:

- Processing time (seconds)
- Number of cells correctly detected as filled
- Cell-by-cell agreement rate between the two methods
- Accuracy against ground truth (if `--ground-truth` is provided)

```bash
# On an existing screenshot (no browser needed)
python benchmark.py --screenshot sudoku_screenshot.png

# With ground truth for accuracy measurement
python benchmark.py \
  --screenshot sudoku_screenshot.png \
  --ground-truth 058030020402000905007000680290054070500620000003810025109003064865049130070000006
```

## Docker

The Dockerfile bundles Chrome + Xvfb for fully headless execution.

```bash
# Build
docker build -t sudoku-solver .

# Run the benchmark on the bundled screenshot (no display required)
docker run --rm sudoku-solver

# Run the full solver with a virtual display
docker run --rm sudoku-solver \
  bash -c "Xvfb :99 -screen 0 1920x1080x24 & sleep 1 && python main.py"
```

---

## Project structure

```
Sudoku-Solver/
├── main.py              # Entry point — full pipeline with CLI flags
├── capture.py           # Selenium: open browser, dismiss popups, screenshot
├── patternMatch.py      # Approach A: OpenCV template matching for digit recognition
├── ocrProcessing.py     # Approach B: EasyOCR for digit recognition
├── benchmark.py         # Timed, quantified comparison of both approaches
├── solver.py            # Pure-Python backtracking solver
├── interact.py          # pyautogui: click + arrow keys + digit entry
├── templates/           # Real digit images (1.png – 9.png) extracted from sudoku.com
│   └── 1.png … 9.png
├── requirements.txt     # Pinned Python dependencies
├── Dockerfile           # Headless Docker image (Chrome + Xvfb)
├── sudoku_screenshot.png  # Sample screenshot (used by benchmark without browser)
└── grid_debug.png       # Debug output: detected grid with overlaid digit labels
```

---

## Digit templates

The `templates/` directory contains real digit images (54×56 px) cropped directly from sudoku.com cells. Using actual site glyphs — rather than synthetically generated fonts — is key to template matching accuracy: the OpenCV `TM_CCOEFF_NORMED` correlation score stays consistently above the 0.55 acceptance threshold across all nine digits.

---

## Key design decisions

**Why classical CV for grid detection (not YOLO)?**
The sudoku grid is always the largest high-contrast rectangle in the screenshot. Canny edge detection + largest-contour selection is robust, instant (< 5 ms), and requires no training data. A learned detector would add complexity without reliability gains for this single, well-defined shape.

**Why real templates instead of synthetic ones?**
Synthetic templates generated with `cv2.putText` use a different font from sudoku.com's CSS-rendered digits. The correlation score between a synthetic template and a real cell glyph falls below threshold for nearly all digits (0/81 accuracy in testing). Real cropped templates achieve 100% accuracy on the same test set.

**Why arrow-key navigation instead of per-cell clicks?**
Clicking each cell triggers sudoku.com's focus handler, which reloads the page on every click. Arrow-key navigation from a single anchor click avoids this and is also faster (no mouse movement latency).
