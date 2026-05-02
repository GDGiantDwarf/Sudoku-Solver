"""
Sudoku solver using the py-sudoku library.

py-sudoku wraps a well-tested constraint-propagation engine. Using it here
deliberately demonstrates the pragmatic engineering choice of relying on a
proven, maintained library rather than reimplementing a known algorithm.

Install: pip install py-sudoku
"""

from sudoku import Sudoku


def solve_sudoku(board):
    """
    Solve a 9x9 sudoku board.

    Args:
        board: 9x9 list of ints; 0 represents an empty cell.

    Returns:
        Solved 9x9 list of ints, or None if the puzzle has no solution.
    """
    puzzle = Sudoku(3, 3, board=board)
    solution = puzzle.solve()
    return solution.board if solution else None
