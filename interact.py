# Module qui saisit automatiquement une solution de sudoku dans le navigateur via les touches fléchées.
# 
# Usage
# 
# from interact import enter_solution
# 
# enter_solution(driver, solved_board, original_board, start_coord)
# driver — WebDriver Selenium (navigateur ouvert sur sudoku.com)
# solved_board — grille résolue 9x9
# original_board — grille originale (0 = case vide)
# start_coord — tuple (x, y) pixel du centre de la cellule (0,0)
# Exemple de données
# 
# original_board = [
#     [5, 3, 0, 0, 7, 0, 0, 0, 0],
#     [6, 0, 0, 1, 9, 5, 0, 0, 0],
#     [0, 9, 8, 0, 0, 0, 0, 6, 0],
#     [8, 0, 0, 0, 6, 0, 0, 0, 3],
#     [4, 0, 0, 8, 0, 3, 0, 0, 1],
#     [7, 0, 0, 0, 2, 0, 0, 0, 6],
#     [0, 6, 0, 0, 0, 0, 2, 8, 0],
#     [0, 0, 0, 4, 1, 9, 0, 0, 5],
#     [0, 0, 0, 0, 8, 0, 0, 7, 9],
# ]
# 
# solved_board = [
#     [5, 3, 4, 6, 7, 8, 9, 1, 2],
#     [6, 7, 2, 1, 9, 5, 3, 4, 8],
#     [1, 9, 8, 3, 4, 2, 5, 6, 7],
#     [8, 5, 9, 7, 6, 1, 4, 2, 3],
#     [4, 2, 6, 8, 5, 3, 7, 9, 1],
#     [7, 1, 3, 9, 2, 4, 8, 5, 6],
#     [9, 6, 1, 5, 3, 7, 2, 8, 4],
#     [2, 8, 7, 4, 1, 9, 6, 3, 5],
#     [3, 4, 5, 2, 8, 6, 1, 7, 9],
# ]



"""
Module d'interaction avec le navigateur.
Clique une seule fois sur la grille pour la focus,
puis navigue avec les touches fléchées pour éviter les rechargements de page.
"""
import time
import pyautogui
from selenium.webdriver.common.keys import Keys


def enter_solution(driver, solved_board, original_board, start_coord, delay=0.08):
    """
    Saisit la solution en naviguant avec les touches fléchées.

    Args:
        driver: instance Selenium WebDriver
        solved_board: grille 9x9 résolue
        original_board: grille 9x9 originale (0 = vide)
        start_coord: tuple (x, y) coordonnées écran du centre de la cellule (0,0)
        delay: délai entre chaque action
    """
    driver.execute_script("window.focus();")
    time.sleep(0.3)

    # Cliquer une seule fois sur la cellule (0,0) pour donner le focus à la grille
    x0, y0 = start_coord
    pyautogui.click(x0, y0)
    time.sleep(0.4)

    current_row, current_col = 0, 0

    for row in range(9):
        for col in range(9):
            if original_board[row][col] != 0:
                continue  # Cellule déjà remplie, on skip

            digit = solved_board[row][col]

            # Naviguer depuis la position actuelle jusqu'à la cible
            _navigate_to(current_row, current_col, row, col, delay)
            current_row, current_col = row, col

            # Saisir le chiffre
            print(f"  Saisie de {digit} en ({row}, {col})...")
            pyautogui.press(str(digit))
            time.sleep(delay)

    print("  ✅ Solution saisie !")


def _navigate_to(from_row, from_col, to_row, to_col, delay):
    """
    Navigue de (from_row, from_col) à (to_row, to_col) avec les touches fléchées.
    """
    row_diff = to_row - from_row
    col_diff = to_col - from_col

    # Déplacements verticaux
    if row_diff > 0:
        for _ in range(row_diff):
            pyautogui.press("down")
            time.sleep(delay)
    elif row_diff < 0:
        for _ in range(-row_diff):
            pyautogui.press("up")
            time.sleep(delay)

    # Déplacements horizontaux
    if col_diff > 0:
        for _ in range(col_diff):
            pyautogui.press("right")
            time.sleep(delay)
    elif col_diff < 0:
        for _ in range(-col_diff):
            pyautogui.press("left")
            time.sleep(delay)


if __name__ == "__main__":
    print("Module d'interaction prêt.")