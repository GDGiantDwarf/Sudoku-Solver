# Sudoku Solver — Computer Vision

A Sudoku solver driven entirely by **computer vision**: the program takes a screenshot of [sudoku.com](https://sudoku.com/), detects the grid and reads the digits visually, solves the puzzle, then either types the solution into the browser or renders it onto the screenshot image — **without reading the HTML DOM** to perceive the grid or its digits (fundamental constraint of the assignment).

---

## Pipeline

| Step | What happens | File |
|------|-------------|------|
| 1 — Screenshot | Selenium opens sudoku.com, dismisses cookie/tutorial popups, takes a full-page screenshot | `capture.py` |
| 2 — Grid detection | Canny edge detection → largest external contour → 4-corner approximation → axis-aligned crop | `gridDetection.py` |
| 3 — Digit recognition | **Approach A**: template matching (`cv2.matchTemplate`) against real templates extracted from sudoku.com. **Approach B**: Tesseract OCR on each cell crop | `patternMatch.py`, `ocrProcessing.py` |
| 4 — Solving | `py-sudoku` constraint-propagation engine | `solver.py` |
| 5a — Interaction *(local)* | Single pyautogui click on cell (0,0) for grid focus, then arrow keys + digit keys to fill every empty cell | `interact.py` |
| 5b — Render *(Docker)* | Solved digits drawn as green overlays directly onto the screenshot → `solved_sudoku.png` | `render_solution.py` |

DOM usage is limited exclusively to closing the cookie banner and tutorial modal (exception explicitly permitted by the assignment).

---

## Tech stack

| Brick | Library | Role |
|-------|---------|------|
| Browser automation | `selenium` + ChromeDriver | Open sudoku.com, screenshot, popup dismissal |
| Grid detection | `opencv-python` (Canny) | Contour detection, 4-corner approximation, axis-aligned crop |
| Grid detection (alt) | `ultralytics` YOLOv8n | Fine-tuned detector — local/benchmark only |
| Digit recognition A | `opencv-python` (`cv2.matchTemplate`) | Template matching against real digit images |
| Digit recognition B | `pytesseract` + Tesseract | OCR on individual cell crops (PSM 6, LSTM) |
| Solving | `py-sudoku` | Constraint-propagation engine |
| Input automation | `pyautogui` | Click + arrow keys + digit key presses — local only |
| Solution rendering | `opencv-python` | Overlay solved digits onto screenshot — Docker only |

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

This installs all dependencies including `ultralytics` (YOLOv8) via the CPU-only PyTorch index.

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
# Solve the current puzzle on sudoku.com (Canny + Template Matching)
python main.py

# Run the benchmark — compare all 4 pipeline combinations on a live puzzle
python main.py --benchmark

# Benchmark repeated over N fresh puzzles (compound statistics)
python main.py --benchmark --runs 10
```

> **Note**: Do not move your mouse during the input phase — `pyautogui` takes full keyboard/mouse control while typing the solution.

---

## Docker

The Docker image runs **headless Chrome** (no display required) and uses a lean dependency set — `ultralytics`, `pyautogui`, and the benchmark are excluded to keep the image small and fast to build.

**What Docker does instead of typing:** since `pyautogui` cannot control a display inside a container, the solver draws the solved digits as green overlays directly onto the screenshot and saves `solved_sudoku.png`.

```bash
# Build (~2 min — downloads Chrome and Tesseract, no PyTorch)
docker build -t sudoku-solver .

# Solve a live puzzle and retrieve the output image
docker run --rm -v "${PWD}:/output" sudoku-solver bash -c "python main.py && cp solved_sudoku.png /output/"

# The solved image is now in your current directory as solved_sudoku.png
```

The container fetches a fresh puzzle from sudoku.com on every run. The full pipeline executes inside the container:

```
headless Chrome → screenshot → Canny → Template Matching → py-sudoku → render_solution.py → solved_sudoku.png
```


---

## Reproducing the YOLO training

The dataset was collected automatically using `collect_dataset.py`, which opens sudoku.com via Selenium, captures screenshots, and auto-annotates the grid bounding box using the same Canny pipeline as production.

```bash
# 1. Collect dataset (opens browser, captures ~80 images)
python collect_dataset.py --n 80 --out dataset

# 2. Train YOLOv8n (50 epochs, CPU or GPU)
python train.py --data dataset/data.yaml --epochs 50 --model yolov8n.pt

# Weights saved to: runs/detect/sudoku_detector/weights/best.pt

# 3. Visually inspect predictions on training images
python test.py --weights runs/detect/sudoku_detector/weights/best.pt --images dataset/images/train
# SPACE = next image, ESC = quit
```

Pre-trained weights are already included at `runs/detect/sudoku_detector/weights/best.pt` (mAP50: 0.995).

---

## Project structure

```
Sudoku-Solver/
├── main.py                # Entry point — local solve, benchmark, or Docker render
├── capture.py             # Selenium: open browser, dismiss popups, screenshot
├── gridDetection.py       # Grid detection: Canny (default) or YOLOv8
├── patternMatch.py        # Approach A: template matching digit recognition
├── ocrProcessing.py       # Approach B: Tesseract digit recognition
├── benchmark.py           # 4-pipeline comparison (Canny/YOLO × TM/Tesseract)
├── solver.py              # py-sudoku constraint-propagation solver
├── interact.py            # pyautogui: anchor click + arrow keys + digit entry
├── render_solution.py     # Docker output: draw solution onto screenshot image
├── collect_dataset.py     # Auto-capture + auto-annotate dataset for YOLO training
├── train.py               # YOLOv8 training script
├── test.py                # Visual bbox inspection of trained model
├── templates/             # Real digit images (1.png – 9.png) from sudoku.com
├── requirements.txt       # Full local dependencies (includes ultralytics)
├── requirements_docker.txt# Lean Docker dependencies (no ultralytics/pyautogui)
├── Dockerfile             # Headless Docker image (Chrome + Tesseract, no PyTorch)
├── sudoku_screenshot.png  # Sample screenshot
└── grid_debug.png         # Debug output: detected grid with overlaid digit labels
```

---

## Digit templates

The `templates/` directory contains real digit images (54×56 px) cropped directly from sudoku.com cells. Using actual site glyphs — rather than synthetically generated fonts — is key to template matching accuracy: the `TM_CCOEFF_NORMED` correlation score stays consistently above the 0.55 acceptance threshold across all nine digits.

---

## Key design decisions

**Why Canny for grid detection in production?**
The sudoku grid is always the largest high-contrast rectangle in the screenshot. Canny + largest-contour selection is robust, instant (0.07–0.10 s), and requires no training data or external dependencies. YOLOv8 is available for the benchmark but Canny was chosen as the production default after 200-run evaluation showed identical accuracy at 2× the speed.

**Why arrow-key navigation instead of per-cell clicks?**
Clicking each cell triggers sudoku.com's focus handler, which reloads the page on every click. Arrow-key navigation from a single anchor click avoids this and is faster (no mouse movement latency).

**Why render to image in Docker instead of interacting?**
`pyautogui` controls the physical mouse and keyboard — it requires a real desktop session. Inside a headless container there is no desktop to control. The render approach produces a verifiable output image without any display dependency.


## Hosting:
This repo is hosted at: https://github.com/GDGiantDwarf/Sudoku-Solver
Yolo weights are hosted at: https://huggingface.co/GDGiantDwarf/SudokuGridDetector/