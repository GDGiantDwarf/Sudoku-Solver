"""
Module d'initialisation du navigateur et capture d'ecran.
Ouvre sudoku.com, ferme le bandeau cookies et la modale tutoriel,
puis prend un screenshot.

Screenshot order:
  1. Dismiss tutorial overlay (covers the grid — must go first)
  2. Take screenshot
  3. Dismiss cookie banner (triggers a page reload; done last so it
     doesn't crash the renderer before we have our image)

Headless mode: set the DOCKER environment variable to run Chrome without
a physical display (required inside a container).
"""

import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


PAGE_LOAD_DELAY = 5


def open_browser_and_screenshot(url, screenshot_path):
    """
    Ouvre sudoku.com, ferme les overlays, prend un screenshot.
    Retourne le driver Selenium (garde ouvert pour l'interaction).
    """
    print(f"\n[1] Ouverture de {url}...")

    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--window-position=0,0")
    # Reduce background activity that can destabilise the renderer
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

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(PAGE_LOAD_DELAY)

    # ── 1. Dismiss tutorial overlay ───────────────────────────────────────────
    # The tutorial overlay covers the grid and darkens the whole page.
    # Must be gone before we take the screenshot.
    _dismiss_tutorial(driver)
    time.sleep(PAGE_LOAD_DELAY)

    # ── 2. Screenshot ─────────────────────────────────────────────────────────
    # Taken before the cookie click: clicking the banner triggers a page
    # reload that can crash the Chrome renderer subprocess.
    driver.save_screenshot(screenshot_path)
    print(f"  Screenshot sauvegarde : {screenshot_path}")

    # ── 3. Cookie banner ──────────────────────────────────────────────────────
    # Dismissed after the screenshot so the browser is in a clean state
    # for the interaction phase (interact.py needs focus on the grid).
    _dismiss_cookie_banner(driver)

    return driver


def _dismiss_tutorial(driver):
    """
    Ferme la modale tutoriel de sudoku.com si presente.
    Essaie le bouton IntroJS, puis un clic JS sur l'overlay, puis Echap.
    """
    # IntroJS skip button (sudoku.com's onboarding library)
    for sel in (".introjs-skipbutton", ".skip-btn", "[data-step-number]"):
        try:
            btn = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            btn.click()
            time.sleep(0.5)
            print(f"  Tutorial ferme : {sel}")
            return
        except Exception:
            pass

    # Click the overlay backdrop (dismisses many modal types)
    try:
        driver.execute_script(
            "document.querySelector('.introjs-overlay')?.click();"
            "document.querySelector('.modal-backdrop')?.click();"
        )
        time.sleep(0.5)
    except Exception:
        pass

    # Press Escape as a last resort
    try:
        from selenium.webdriver.common.keys import Keys
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(0.5)
    except Exception:
        pass


def _dismiss_cookie_banner(driver):
    """
    Clique sur le bouton d'acceptation des cookies OneTrust si present.
    Attend jusqu'a 5 s; si absent ou si Chrome devient non-reactif, continue.
    """
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))
        )
        btn.click()
        time.sleep(2)
        print("  Cookie banner ferme.")
    except Exception:
        pass


if __name__ == "__main__":
    url = "https://sudoku.com/"
    screenshot_path = "sudoku_screenshot.png"
    open_browser_and_screenshot(url, screenshot_path)

def navigate_and_screenshot(driver, url, screenshot_path, retries=15, retry_delay=10):
    """
    Navigate to a fresh puzzle in an already-open browser and take a screenshot.
    Used by multi-run benchmark mode to get a new puzzle without reopening Chrome.
    Retries on transient network errors (e.g. ERR_NAME_NOT_RESOLVED).
    """
    import time
    from selenium.common.exceptions import WebDriverException
    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            time.sleep(PAGE_LOAD_DELAY)
            _dismiss_tutorial(driver)
            time.sleep(PAGE_LOAD_DELAY)
            driver.save_screenshot(screenshot_path)
            print(f"  Screenshot sauvegarde : {screenshot_path}")
            _dismiss_cookie_banner(driver)
            return
        except WebDriverException as e:
            if attempt == retries:
                raise
            print(f"  Network error (attempt {attempt}/{retries}): {e.msg.splitlines()[0]} — retrying in {retry_delay}s...")
            time.sleep(retry_delay)