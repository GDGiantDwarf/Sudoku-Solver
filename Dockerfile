# ── Sudoku Solver — Docker image ──────────────────────────────────────────────
#
# Solve mode: headless Chrome fetches a live sudoku.com puzzle, the CV pipeline
# detects + solves it, and the result is rendered onto solved_sudoku.png.
# Benchmark and YOLO are not available in Docker (ultralytics excluded for size).
#
# Build
#   docker build -t sudoku-solver .
#
# Solve a live puzzle and retrieve the output image (PowerShell):
#   docker run --rm -v "${PWD}:/output" sudoku-solver bash -c "python main.py && cp solved_sudoku.png /output/"
#
# Solve a live puzzle (bash / macOS / Linux):
#   docker run --rm -v "$(pwd):/output" sudoku-solver bash -c "python main.py && cp solved_sudoku.png /output/"

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV runtime
        libgl1 \
        libglib2.0-0 \
        # Tesseract OCR engine (pytesseract is just a wrapper around this binary)
        tesseract-ocr \
        # Required to add the Chrome apt repository
        wget \
        ca-certificates \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome (stable) via the official Google apt repository.
# Selenium 4.6+ includes selenium-manager which automatically downloads
# a matching ChromeDriver — no manual ChromeDriver installation needed.
RUN wget -q -O /usr/share/keyrings/google-chrome.gpg \
        https://dl.google.com/linux/linux_signing_key.pub \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] \
        http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# Tell capture.py to run Chrome in headless mode (no physical display in container)
ENV DOCKER=1

# Default: benchmark on the bundled screenshot — proves the CV pipeline works
# without needing a network connection or a running browser.
CMD ["python", "benchmark.py", "--screenshot", "sudoku_screenshot.png"]