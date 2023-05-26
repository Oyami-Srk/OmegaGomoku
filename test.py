from OmegaGomoku import *
import re

board_status = """
    0 1 2 3 4 5 6
0   . . . . X O .
1   . . . O O . .
2   . O . X O . O
3   . . X X O . .
4   . . X X O . .
5   . X X X X . .
6   . . . O . . .


"""

action = (5, 5)


# ['.', 'O', 'X']
def set_board_to_status(board: Board, status: str):
    status = re.sub(r'(\d| )', '', status).strip().split('\n')
    board_size = len(status)
    for y in range(board_size):
        for x in range(board_size):
            if status[y][x] != '.':
                board.put(x, y, ['.', 'O', 'X'].index(status[y][x]))


board = Board(board_size=7)
set_board_to_status(board, board_status)
print(board)
print(board.put(5, 5, Player.BLACK))
print(board)
