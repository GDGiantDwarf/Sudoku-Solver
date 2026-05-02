"""
Dataset collection script for YOLO sudoku grid detection.

Opens sudoku.com in a loop, takes screenshots, auto-labels the grid bounding
box using the same CV pipeline used in production (Canny -> largest contour ->
4-corner approximation), then writes image/label pairs in YOLO format.

Output layout
-------------
dataset/
  images/train/   75% of captures
  images/val/     25% of captures
  labels/train/
  labels/val/
  data.yaml       ready for `yolo train data=dataset/data.yaml ...`

Usage
-----
    python collect_dataset.py               # 80 images
    python collect_dataset.py --n 120       # collect more
    python collect_dataset.py --out mydata  # different output folder
"""

import argparse
import os
import random
import time

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from patternMatch import order_points

VAL_RATIO = 0.25
CLASS_ID  = 0
URL       = "https://sudoku.com/"

WINDOW_SIZES = [
    (1920, 1080),
    (1440, 900),
    (1280, 800),
    (1600, 1000),
]


def _open_browser(window_size=(1920, 1080)):
    options = Options()
    w, h = window_size
    options.add_argument(f"--window-size={w},{h}")
    options.add_argument("--window-position=0,0")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-sync")
    options.add_argument("--no-first-run")
    options.add_argument("--disable-features=OptimizationGuideModelDownloading,TranslateUI")
    options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2
    })
    if os.environ.get("DOCKER"):
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def _dismiss_tutorial(driver):
    for sel in (".introjs-skipbutton", ".skip-btn", "[data-step-number]"):
        try:
            btn = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            btn.click()
            time.sleep(0.5)
            return
        except Exception:
            pass
    try:
        driver.execute_script(
            "document.querySelector('.introjs-overlay')?.click();"
            "document.querySelector('.modal-backdrop')?.click();"
        )
        time.sleep(0.5)
    except Exception:
        pass
    try:
        from selenium.webdriver.common.keys import Keys
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(0.3)
    except Exception:
        pass


def _load_fresh_puzzle(driver):
    driver.get(URL)
    time.sleep(4)
    _dismiss_tutorial(driver)
    time.sleep(1)


def _detect_grid_bbox(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest  = max(contours, key=cv2.contourArea)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    if cv2.contourArea(largest) < 0.05 * img_area:
        return None

    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) != 4:
        return None

    pts = order_points(approx.reshape(4, 2))
    xs, ys = pts[:, 0], pts[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def _save_pair(img_bgr, bbox, stem, split, out_dir):
    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cv2.imwrite(os.path.join(img_dir, f"{stem}.png"), img_bgr)

    h, w = img_bgr.shape[:2]
    cx, cy, bw, bh = _to_yolo(bbox, w, h)
    with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
        f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _write_yaml(out_dir):
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"""path: {os.path.abspath(out_dir)}
train: images/train
val:   images/val
nc: 1
names: [sudoku_grid]
""")


def collect(n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    splits = ["val" if random.random() < VAL_RATIO else "train"
              for _ in range(n)]

    saved   = 0
    skipped = 0
    driver  = None
    current_window = None

    print(f"Collecting {n} images -> {out_dir}/")
    print(f"Train: {splits.count('train')}  Val: {splits.count('val')}\n")

    try:
        for i in range(n):
            window_size = random.choice(WINDOW_SIZES)
            split       = splits[i]
            stem        = f"sudoku_{i+1:04d}"

            if window_size != current_window:
                if driver:
                    driver.quit()
                print(f"  Browser -> {window_size[0]}x{window_size[1]}")
                driver = _open_browser(window_size)
                current_window = window_size

            print(f"  [{i+1:3d}/{n}] {split:<6} ", end="", flush=True)

            _load_fresh_puzzle(driver)

            png_bytes = driver.get_screenshot_as_png()
            img       = cv2.imdecode(np.frombuffer(png_bytes, np.uint8),
                                     cv2.IMREAD_COLOR)

            bbox = _detect_grid_bbox(img)
            if bbox is None:
                print("SKIP (grid not detected)")
                skipped += 1
                continue

            _save_pair(img, bbox, stem, split, out_dir)
            print(f"OK  {bbox}")
            saved += 1

    finally:
        if driver:
            driver.quit()

    _write_yaml(out_dir)
    print(f"\nDone. Saved: {saved}  Skipped: {skipped}")
    print(f"Train: {len(os.listdir(os.path.join(out_dir,'images','train')))}")
    print(f"Val:   {len(os.listdir(os.path.join(out_dir,'images','val')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=80,       help="Number of images (default: 80)")
    parser.add_argument("--out",  type=str, default="dataset", help="Output directory (default: dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    collect(args.n, args.out)