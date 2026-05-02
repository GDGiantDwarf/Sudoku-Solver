# ── Sudoku Solver — Docker image ──────────────────────────────────────────────
#
# Supports benchmark mode only (the solve mode uses pyautogui to control a
# real desktop browser and must run on the host machine).
#
# Build
#   docker build -t sudoku-solver .
#
# Run benchmark on the bundled screenshot (no internet needed)
#   docker run --rm sudoku-solver
#
# Run benchmark against a live sudoku.com puzzle (fetches a real page)
#   docker run --rm sudoku-solver python main.py --benchmark
#
# Run with a ground-truth string to measure accuracy
#   docker run --rm sudoku-solver python main.py --benchmark \
#     --ground-truth 058030020402000905007000680290054070500620000003810025109003064865049130070000006

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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# Tell capture.py to run Chrome in headless mode (no physical display in container)
ENV DOCKER=1

# Default: benchmark on the bundled screenshot — proves the CV pipeline works
# without needing a network connection or a running browser.
CMD ["python", "benchmark.py", "--screenshot", "sudoku_screenshot.png"]
