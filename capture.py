"""
Module d'initialisation du navigateur et capture d'écran.
Ouvre sudoku.com, ferme les popups/cookies, et prend un screenshot.
"""

import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


PAGE_LOAD_DELAY = 5


def open_browser_and_screenshot(url, screenshot_path):
    """
    Ouvre sudoku.com et prend un screenshot.
    Retourne le driver Selenium (gardé ouvert pour l'interaction).
    """
    print(f"\n[1/6] Ouverture de {url}...")

    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--window-position=0,0")
    # Accepter les cookies automatiquement si possible
    options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2
    })

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(PAGE_LOAD_DELAY)

    # Fermer les popups/cookies si présents (toléré par le sujet)
    _dismiss_popups(driver)

    driver.save_screenshot(screenshot_path)
    print(f"  Screenshot sauvegardé : {screenshot_path}")

    return driver


def _dismiss_popups(driver):
    """
    Ferme les bandeaux cookies et popups (interaction HTML tolérée par le sujet).
    Attend activement que chaque élément soit cliquable avant de fermer.
    """
    # Selectors connus pour sudoku.com et bandeaux cookies génériques
    selectors = [
        "#onetrust-accept-btn-handler",
        ".cc-accept",
        "[data-testid='cookie-accept']",
        ".modal-close",
        ".close-modal",
        ".popup-close",
        "button[class*='close']",
        ".introjs-skipbutton",   # tutoriel sudoku.com
        ".skip-btn",
    ]

    for sel in selectors:
        try:
            btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            btn.click()
            time.sleep(0.8)
            print(f"  Popup fermé : {sel}")
        except Exception:
            pass

    # Attendre que les overlays aient disparu
    overlay_selectors = [
        ".modal", ".popup", ".overlay",
        "#onetrust-banner-sdk", ".cc-window",
        ".introjs-overlay",
    ]
    for sel in overlay_selectors:
        try:
            WebDriverWait(driver, 5).until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, sel))
            )
        except Exception:
            pass
    time.sleep(1)  # Attente animations de fermeture

if __name__ == "__main__":
    url = "https://sudoku.com/"
    screenshot_path = "sudoku_screenshot.png"
    open_browser_and_screenshot(url, screenshot_path)