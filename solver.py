from sudoku import Sudoku

# board = [
#     [6,0,0,1,0,0,0,0,2],
#     [8,0,1,0,9,0,0,0,0],
#     [0,7,5,0,8,4,0,0,0],
#     [4,3,0,0,2,0,5,6,1],
#     [5,1,8,7,0,0,4,0,9],
#     [0,9,6,4,1,0,3,0,0],
#     [0,0,0,0,7,0,0,0,0],
#     [0,6,0,0,3,1,0,5,0],
#     [7,0,2,5,4,0,6,0,3]
# ]

# puzzle = Sudoku(3, 3, board)
# print("Original Puzzle:")
# print(puzzle)
# solution =puzzle.solve()
# print("Solved Puzzle:")
# solution.show()
# print("Solution Board:")
# print(solution.board)

def solve_sudoku(board):
    """
    Résout une grille de Sudoku donnée sous forme de liste de listes.
    Les cases vides sont représentées par 0.
    Retourne la grille résolue ou None si aucune solution n'existe.
    """
    puzzle = Sudoku(3, 3, board)
    solution = puzzle.solve()
    solution.show()  # Affiche la solution dans la console
    return solution.board if solution else None