import capture
import interact
import solver
import processor

def main():
    url = "https://sudoku.com/"
    screenshot_path = "sudoku_screenshot.png"

    # Étape 1 : Ouvrir le navigateur et faire une capture d'écran
    driver = capture.open_browser_and_screenshot(url, screenshot_path)

    # Étape 2 : Analyser la capture d'écran pour extraire le plateau de jeu
    original_board, start_coord = processor.extract_board_from_screenshot(screenshot_path)

    # Étape 3 : Résoudre le Sudoku
    solved_board = solver.solve_sudoku(original_board)

    # Étape 4 : Interagir avec la page pour saisir la solution
    interact.enter_solution(driver, solved_board, original_board, start_coord)

if __name__ == "__main__":
    main()